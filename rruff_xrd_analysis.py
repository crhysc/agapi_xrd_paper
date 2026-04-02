"""
rruff_xrd_analysis.py — RRUFF Powder XRD → AtomGPT.org Structure Identification

Usage:
  python rruff_xrd_analysis.py
  python rruff_xrd_analysis.py --max-entries 50 --method both --delay 3
  python rruff_xrd_analysis.py --max-abc 10   # only minerals with a,b,c ≤ 10 Å

Jupyter:
  cfg = dict(DEFAULTS)
  cfg["max_abc"] = 10
  cfg["max_entries"] = 50
  df, consolidated = run_pipeline(cfg)
"""

import argparse, json, re, time, sys, os
import numpy as np
import pandas as pd
import requests
from jarvis.core.composition import Composition
from jarvis.db.figshare import data as figshare_data

DEFAULTS = dict(
    base_url="https://atomgpt.org",
    api_key="sk-",
    method="both", wavelength=1.54184, max_entries=1000,
    max_abc=10,  # Max RRUFF a,b,c filter (Å) — None = no filter
    fallback_to_elements=True, run_refinement=False,
    refinement_engine="gsas2", preferred_orientation=False,
    run_alignn=False, run_slakonet=False, delay=2.0,
    max_retries=5, initial_backoff=5.0, backoff_multiplier=2.0,
    max_backoff=120.0, output_csv="rruff_xrd_results_cod_max10.csv",
    output_poscars="rruff_matched_poscars_cod_max10.json",
    cache_dir=".rruff_cache", resume=True, overwrite_cache=False,
    write_every=1,
)

_VALID_ELEMENTS = {
    "H","He","Li","Be","B","C","N","O","F","Ne","Na","Mg","Al","Si","P",
    "S","Cl","Ar","K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu",
    "Zn","Ga","Ge","As","Se","Br","Kr","Rb","Sr","Y","Zr","Nb","Mo",
    "Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe","Cs",
    "Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er",
    "Tm","Yb","Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl",
    "Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th","Pa","U","Np","Pu",
}

def _json_default(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if hasattr(obj, "to_dict"): return obj.to_dict()
    return str(obj)

def format_xrd_data(x_arr, y_arr):
    return "\n".join(f"{xi:.5f} {yi:.6f}" for xi, yi in zip(x_arr, y_arr))

def _get_formula(line):
    if not line or not isinstance(line, str): return None
    line = line.strip()
    if not line: return None
    line = re.sub(r"\[.*?\]", "", line)
    line = re.sub(r"\^[^^]*\^", "", line)
    line = line.replace("_", "")
    line = re.sub(r"(\d+\.?\d*)-\d+\.?\d*", r"\1", line)
    line = line.replace(" ", "")
    line = re.sub(r"[;·][\dxn]*\.?\d*H2O", "", line)
    return line or None

def _get_elements(formula):
    if not formula: return []
    try: return sorted(Composition.from_string(formula).to_dict().keys())
    except Exception: pass
    raw = set(re.findall(r"[A-Z][a-z]?", formula))
    return sorted(raw & _VALID_ELEMENTS)

def parse_rruff_cell_params(cell_str):
    if not cell_str or not isinstance(cell_str, str): return None
    try:
        result = {}
        for key in ["a","b","c","alpha","beta","gamma","volume"]:
            m = re.search(rf"{key}:\s*([\d.]+)", cell_str)
            if m: result[key] = float(m.group(1))
        m = re.search(r"crystal system:\s*(\w+)", cell_str)
        if m: result["crystal_system"] = m.group(1)
        if all(k in result for k in ["a","b","c"]):
            for ang in ["alpha","beta","gamma"]:
                if ang not in result: result[ang] = 90.0
            return result
        return None
    except Exception: return None

def api_post_with_retry(url, payload, headers, cfg):
    backoff = cfg["initial_backoff"]
    for attempt in range(cfg["max_retries"] + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=300)
        if resp.status_code != 429:
            resp.raise_for_status()
            return resp.json()
        retry_after = resp.headers.get("Retry-After")
        wait = min(float(retry_after) if retry_after else backoff, cfg["max_backoff"])
        if attempt < cfg["max_retries"]:
            print(f"    ⏳ Rate limited (429). Waiting {wait:.0f}s (retry {attempt+1}/{cfg['max_retries']})...")
            time.sleep(wait)
            backoff = min(backoff * cfg["backoff_multiplier"], cfg["max_backoff"])
        else:
            resp.raise_for_status()

def call_analyze(formula, xrd_data_str, headers, cfg, **ov):
    payload = {
        "formula": formula, "xrd_data": xrd_data_str, "best_match_poscar": "",
        "wavelength": ov.get("wavelength", cfg["wavelength"]),
        "method": ov.get("method", cfg["method"]),
        "run_refinement": ov.get("run_refinement", cfg["run_refinement"]),
        "refinement_engine": ov.get("refinement_engine", cfg["refinement_engine"]),
        "preferred_orientation": ov.get("preferred_orientation", cfg["preferred_orientation"]),
        "run_alignn": ov.get("run_alignn", cfg["run_alignn"]),
        "run_slakonet": ov.get("run_slakonet", cfg["run_slakonet"]),
    }
    return api_post_with_retry(f"{cfg['base_url']}/xrd/analyze_with_refinement", payload, headers, cfg)

def extract_best_result(result):
    info = {
        "pm_success":False,"pm_jid":None,"pm_similarity":None,"pm_formula":None,
        "pm_spacegroup":None,"pm_num_atoms":None,"pm_top5":[],
        "dg_success":False,"dg_similarity":None,"dg_formula":None,
        "dg_spacegroup":None,"dg_num_atoms":None,
        "refinement_rwp":None,"refinement_engine":None,
        "best_poscar":None,"best_method":None,"best_similarity":None,
    }
    best_sim, best_poscar, best_method = -1, None, None

    pm = result.get("pattern_matching", {})
    if pm.get("success") and pm.get("best_match"):
        bm = pm["best_match"]
        info.update({"pm_success":True,"pm_jid":bm.get("jid"),"pm_similarity":bm.get("similarity"),
            "pm_formula":bm.get("formula"),"pm_spacegroup":bm.get("spacegroup"),
            "pm_num_atoms":bm.get("num_atoms"),"pm_top5":pm.get("top_5_matches",[])})
        if bm.get("similarity",0) > best_sim:
            best_sim, best_poscar, best_method = bm["similarity"], bm.get("poscar"), "pattern_matching"

    dg = result.get("diffractgpt", {})
    if dg.get("success") and dg.get("structure"):
        st = dg["structure"]
        info.update({"dg_success":True,"dg_similarity":st.get("similarity"),
            "dg_formula":st.get("formula"),"dg_spacegroup":st.get("spacegroup"),
            "dg_num_atoms":st.get("num_atoms")})
        if st.get("similarity",0) > best_sim:
            best_sim, best_poscar, best_method = st["similarity"], st.get("poscar"), "diffractgpt"

    ref = result.get("refinement", {})
    if ref and not ref.get("error"):
        info["refinement_rwp"] = ref.get("rwp")
        info["refinement_engine"] = ref.get("engine")
        if ref.get("refined_poscar"): best_poscar = ref["refined_poscar"]

    info["best_poscar"] = best_poscar
    info["best_method"] = best_method
    info["best_similarity"] = best_sim if best_sim > 0 else None
    info["pm_poscar"] = pm.get("best_match",{}).get("poscar") if pm.get("success") else None
    info["dg_poscar"] = dg.get("structure",{}).get("poscar") if dg.get("success") else None
    return info

def poscar_to_atoms(poscar_str):
    if not poscar_str: return None
    try:
        from jarvis.io.vasp.inputs import Poscar
        return Poscar.from_string(poscar_str).atoms
    except Exception: return None

def _get_cell_variants(atoms):
    variants = [("original", atoms)]
    try:
        prim = atoms.get_primitive_atoms
        if prim is not None and prim.num_atoms > 0: variants.append(("primitive", prim))
    except Exception: pass
    try:
        conv = atoms.get_conventional_atoms
        if conv is not None and conv.num_atoms > 0: variants.append(("conventional", conv))
    except Exception: pass
    return variants

def best_cell_match(atoms, rruff_cell):
    if atoms is None or rruff_cell is None:
        if atoms is not None:
            lat = atoms.lattice
            return atoms, "original", {"a":lat.a,"b":lat.b,"c":lat.c,"alpha":lat.alpha,"beta":lat.beta,"gamma":lat.gamma,"volume":lat.volume,"cell_type":"original"}
        return None, None, None

    from itertools import permutations
    exp_a,exp_b,exp_c = rruff_cell["a"],rruff_cell["b"],rruff_cell["c"]
    exp_alpha,exp_beta,exp_gamma = rruff_cell.get("alpha",90.0),rruff_cell.get("beta",90.0),rruff_cell.get("gamma",90.0)
    exp_vol = rruff_cell.get("volume",0)

    best_score, best_result, best_atoms_out, best_label = float("inf"), None, None, "original"
    for var_label, var_atoms in _get_cell_variants(atoms):
        lat = var_atoms.lattice
        abc, angles = [lat.a,lat.b,lat.c], [lat.alpha,lat.beta,lat.gamma]
        for perm in permutations(range(3)):
            pa,pb,pc = abc[perm[0]],abc[perm[1]],abc[perm[2]]
            angle_map = {(0,1):angles[2],(0,2):angles[1],(1,2):angles[0]}
            def _ga(i,j): return angle_map.get((min(i,j),max(i,j)),90.0)
            p_alpha,p_beta,p_gamma = _ga(perm[1],perm[2]),_ga(perm[0],perm[2]),_ga(perm[0],perm[1])
            len_err = abs(pa-exp_a)/max(exp_a,0.1)+abs(pb-exp_b)/max(exp_b,0.1)+abs(pc-exp_c)/max(exp_c,0.1)
            ang_err = abs(p_alpha-exp_alpha)/90+abs(p_beta-exp_beta)/90+abs(p_gamma-exp_gamma)/90
            vol_err = abs(lat.volume-exp_vol)/max(exp_vol,1.0) if exp_vol>0 else 0
            score = len_err + ang_err + 0.5*vol_err
            if score < best_score:
                best_score = score
                best_result = {"a":pa,"b":pb,"c":pc,"alpha":p_alpha,"beta":p_beta,"gamma":p_gamma,"volume":lat.volume,"cell_type":var_label}
                best_atoms_out, best_label = var_atoms, var_label
    return best_atoms_out, best_label, best_result

def print_cell_comparison(rruff_cell, matched_params, cell_label="original"):
    if rruff_cell is None or matched_params is None: return
    print(f"  ── Lattice: Target (RRUFF) vs Predicted ({cell_label}) ──")
    print(f"     {'Param':>7s}   {'Target':>10s}   {'Predicted':>10s}   {'Δ':>10s}")
    for key in ["a","b","c","alpha","beta","gamma"]:
        t,p = rruff_cell.get(key), matched_params.get(key)
        if t is not None and p is not None:
            print(f"     {key:>7s}   {t:10.4f}   {p:10.4f}   {p-t:+10.4f}")
    tv,pv = rruff_cell.get("volume"), matched_params.get("volume")
    if tv is not None and pv is not None:
        print(f"     {'vol':>7s}   {tv:10.2f}   {pv:10.2f}   {pv-tv:+10.2f}")
    cs = rruff_cell.get("crystal_system")
    if cs: print(f"     Crystal system (RRUFF): {cs}")

def print_structure(info, label=""):
    prefix = f"  {label}" if label else "  "
    if info["pm_success"]:
        print(f"{prefix}── Pattern Match: {info['pm_jid']} ──")
        print(f"{prefix}   Formula:    {info['pm_formula']}")
        print(f"{prefix}   Spacegroup: {info['pm_spacegroup']}")
        print(f"{prefix}   Num atoms:  {info['pm_num_atoms']}")
        print(f"{prefix}   Similarity: {info['pm_similarity']:.4f}")
        top5 = info.get("pm_top5", [])
        if top5 and len(top5) > 1:
            print(f"{prefix}   Top-5 matches:")
            for i, m in enumerate(top5):
                print(f"{prefix}     {i+1}. {m.get('jid','?'):16s} {m.get('formula','?'):12s} SG={m.get('spacegroup','?'):8s} sim={m.get('similarity',0):.4f}")
    if info["dg_success"]:
        print(f"{prefix}── DiffractGPT ──")
        print(f"{prefix}   Formula:    {info['dg_formula']}")
        print(f"{prefix}   Spacegroup: {info['dg_spacegroup']}")
        print(f"{prefix}   Num atoms:  {info['dg_num_atoms']}")
        print(f"{prefix}   Similarity: {info['dg_similarity']:.4f}")
    poscar = info.get("best_poscar")
    if poscar:
        lines = poscar.strip().splitlines()
        n_show = min(15, len(lines))
        print(f"{prefix}── Best POSCAR ({info['best_method']}) ──")
        for line in lines[:n_show]: print(f"{prefix}   {line}")
        if len(lines) > n_show: print(f"{prefix}   ... ({len(lines)-n_show} more lines)")
        atoms = poscar_to_atoms(poscar)
        if atoms is not None:
            lat = atoms.lattice
            print(f"{prefix}   Lattice: a={lat.a:.4f} b={lat.b:.4f} c={lat.c:.4f} α={lat.alpha:.2f} β={lat.beta:.2f} γ={lat.gamma:.2f} V={lat.volume:.2f} ų")
            print(f"{prefix}   Elements: {sorted(set(atoms.elements))}")
            print(f"{prefix}   Density:  {atoms.density:.4f} g/cm³")


def _entry_uid(entry):
    rid = str(entry.get("##RRUFFID", "unknown"))
    name = re.sub(r"[^A-Za-z0-9_-]", "_", str(entry.get("##NAMES", "unknown")))
    return f"{rid}__{name}"


def _row_cacheable(row):
    out = {}
    for k, v in row.items():
        if k == "atoms":
            out[k] = None
        elif k in ("pm_atoms_dict", "dg_atoms_dict", "best_atoms_dict") and v is not None and hasattr(v, "to_dict"):
            out[k] = v.to_dict()
        else:
            out[k] = v
    return out


def _entry_payload_from_row(r):
    return {
        "mineral_name":r.get("mineral_name"),"rruff_id":r.get("rruff_id"),"formula":r.get("formula"),
        "elements":r.get("elements"),"query_used":r.get("query_used"),"num_points":r.get("num_points"),
        "rruff_cell":{"a":r.get("rruff_a"),"b":r.get("rruff_b"),"c":r.get("rruff_c"),
            "alpha":r.get("rruff_alpha"),"beta":r.get("rruff_beta"),"gamma":r.get("rruff_gamma"),
            "volume":r.get("rruff_volume"),"crystal_system":r.get("rruff_crystal_system")},
        "pattern_matching":{"success":bool(r.get("pm_success")),"jid":r.get("pm_jid"),
            "similarity":r.get("pm_similarity"),"formula":r.get("pm_formula"),
            "spacegroup":r.get("pm_spacegroup"),"num_atoms":r.get("pm_num_atoms"),
            "poscar":r.get("pm_poscar"),"atoms_dict":r.get("pm_atoms_dict")},
        "diffractgpt":{"success":bool(r.get("dg_success")),"similarity":r.get("dg_similarity"),
            "formula":r.get("dg_formula"),"spacegroup":r.get("dg_spacegroup"),
            "num_atoms":r.get("dg_num_atoms"),"poscar":r.get("dg_poscar"),"atoms_dict":r.get("dg_atoms_dict")},
        "best_match":{"method":r.get("best_method"),"similarity":r.get("best_similarity"),
            "poscar":r.get("best_poscar"),"atoms_dict":r.get("best_atoms_dict")},
        "predicted_cell":{"a":r.get("pred_a"),"b":r.get("pred_b"),"c":r.get("pred_c"),
            "alpha":r.get("pred_alpha"),"beta":r.get("pred_beta"),"gamma":r.get("pred_gamma"),
            "volume":r.get("pred_volume"),"cell_type":r.get("cell_type")},
        "refinement":{"rwp":r.get("refinement_rwp"),"engine":r.get("refinement_engine")},
        "time_s":r.get("time_s"),"error":r.get("error"),
    }


def _atomic_write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _atomic_write_csv(path, df):
    tmp = path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _load_cached_results(cache_dir):
    cache = {}
    if not os.path.isdir(cache_dir):
        return cache
    for fn in sorted(os.listdir(cache_dir)):
        if not fn.endswith(".json"):
            continue
        fp = os.path.join(cache_dir, fn)
        try:
            with open(fp) as f:
                payload = json.load(f)
            row = payload.get("row") if isinstance(payload, dict) else None
            entry_id = payload.get("entry_id") if isinstance(payload, dict) else None
            if row is not None and entry_id is not None:
                cache[entry_id] = row
        except Exception as e:
            print(f"  ⚠ Could not read cache file {fp}: {e}")
    return cache


def _checkpoint_row(row, cache_dir):
    entry_id = row.get("entry_id")
    if not entry_id:
        return
    payload = {"entry_id": entry_id, "row": _row_cacheable(row)}
    _atomic_write_json(os.path.join(cache_dir, f"{entry_id}.json"), payload)


def _write_live_outputs(results, cfg, work_dir):
    if not results:
        return
    df_live = pd.DataFrame(results)
    out_cols = [c for c in df_live.columns if c not in ("best_poscar","pm_poscar","dg_poscar","atoms","pm_top5","pm_atoms_dict","dg_atoms_dict","best_atoms_dict")]
    _atomic_write_csv(cfg["output_csv"], df_live[out_cols])

    live_json = os.path.join(work_dir, "all_results_live.json")
    _atomic_write_json(live_json, [_entry_payload_from_row(r) for r in results])

    poscar_mask = df_live["best_poscar"].fillna("").astype(str).str.strip().ne("") if "best_poscar" in df_live.columns else pd.Series(False, index=df_live.index)
    poscar_entries = df_live.loc[poscar_mask]
    if len(poscar_entries) > 0:
        poscar_dict = {f"{r['mineral_name']}_{r['formula']}":r["best_poscar"] for _,r in poscar_entries.iterrows()}
        _atomic_write_json(cfg["output_poscars"], poscar_dict)


def _save_final_results(df, cfg):
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"rruff_results_{timestamp}"
    entries_dir = os.path.join(results_dir, "entries")
    os.makedirs(entries_dir, exist_ok=True)
    print(f"\n{'═'*60}\nSAVING RESULTS → {results_dir}/\n{'═'*60}")

    consolidated = []
    for _, r in df.iterrows():
        name = r["mineral_name"]; safe_name = re.sub(r"[^A-Za-z0-9_-]","_",name)
        entry_data = _entry_payload_from_row(r)
        with open(os.path.join(entries_dir, f"{safe_name}.json"),"w") as f:
            json.dump(entry_data, f, indent=2, default=_json_default)
        consolidated.append(entry_data)

    print(f"✓ {len(consolidated)} per-entry JSONs → {entries_dir}/")

    consolidated_path = os.path.join(results_dir, "all_results.json")
    with open(consolidated_path,"w") as f: json.dump(consolidated, f, indent=2, default=_json_default)
    print(f"✓ Consolidated JSON → {consolidated_path}")

    csv_path = os.path.join(results_dir, "results.csv")
    out_cols = [c for c in df.columns if c not in ("best_poscar","pm_poscar","dg_poscar","atoms","pm_top5","pm_atoms_dict","dg_atoms_dict","best_atoms_dict")]
    df[out_cols].to_csv(csv_path, index=False)
    print(f"✓ CSV summary → {csv_path}")
    df[out_cols].to_csv(cfg["output_csv"], index=False)

    poscar_mask = df["best_poscar"].fillna("").astype(str).str.strip().ne("") if "best_poscar" in df.columns else pd.Series(False, index=df.index)
    poscar_entries = df.loc[poscar_mask]
    if len(poscar_entries)>0:
        poscar_dict = {f"{r['mineral_name']}_{r['formula']}":r["best_poscar"] for _,r in poscar_entries.iterrows()}
        with open(cfg["output_poscars"],"w") as f: json.dump(poscar_dict, f, indent=2)

    cfg_plots = dict(cfg); cfg_plots["output_csv"] = csv_path
    lattice_comparison(df, cfg_plots)

    n_pm = sum(1 for e in consolidated if e["pattern_matching"]["atoms_dict"])
    n_dg = sum(1 for e in consolidated if e["diffractgpt"]["atoms_dict"])
    n_best = sum(1 for e in consolidated if e["best_match"]["atoms_dict"])
    print(f"\n✓ Atoms dicts saved: PM={n_pm}, DG={n_dg}, Best={n_best}")
    print(f"✓ Results folder: {os.path.abspath(results_dir)}/")
    print(
        f"\n  To load:\n"
        f"    import json; from jarvis.core.atoms import Atoms\n"
        f"    with open('{consolidated_path}') as f: data = json.load(f)\n"
        f"    atoms = Atoms.from_dict(data[0]['best_match']['atoms_dict'])\n"
    )

    return results_dir, consolidated, consolidated_path


# ═══════════════════════════════════════════════════════════════════════════
def process_entry(entry, headers, cfg):
    name = entry["##NAMES"]
    formula = _get_formula(entry["##IDEAL CHEMISTRY"])
    elements = _get_elements(formula)
    xrd_str = format_xrd_data(entry["x"], entry["y"])

    row = {"mineral_name":name,"rruff_id":entry["##RRUFFID"],"formula":formula,
           "elements":",".join(elements),"num_points":len(entry["x"]),"query_used":formula,"error":None}

    rruff_cell = parse_rruff_cell_params(entry.get("##CELL PARAMETERS",""))
    if rruff_cell:
        for k in ["a","b","c","alpha","beta","gamma","volume"]: row[f"rruff_{k}"] = rruff_cell.get(k)
        row["rruff_crystal_system"] = rruff_cell.get("crystal_system")

    try:
        t0 = time.time()
        result = call_analyze(formula, xrd_str, headers, cfg)
        info = extract_best_result(result)

        # Fallback to elements
        if cfg["fallback_to_elements"] and not info["pm_success"] and elements:
            elem_query = ",".join(elements)
            print(f"  → Formula match failed, retrying with elements: {elem_query}")
            row["query_used"] = f"elements:{elem_query}"
            time.sleep(cfg["delay"])
            result2 = call_analyze(elem_query, xrd_str, headers, cfg,
                method="pattern_matching", run_refinement=False, run_alignn=False, run_slakonet=False)
            info2 = extract_best_result(result2)
            if info2["pm_success"] and (not info["pm_success"] or (info2["pm_similarity"] or 0) > (info["pm_similarity"] or 0)):
                for k in ["pm_success","pm_jid","pm_similarity","pm_formula","pm_spacegroup","pm_num_atoms","pm_poscar"]:
                    info[k] = info2[k]
                if (info2["pm_similarity"] or 0) > (info.get("best_similarity") or 0):
                    info["best_similarity"] = info2["pm_similarity"]
                    info["best_method"] = "pattern_matching (elements)"
                    if info2["best_poscar"]: info["best_poscar"] = info2["best_poscar"]
                print(f"  → Element match: {info2['pm_jid']} (sim={info2['pm_similarity']:.3f})")

        elapsed = time.time() - t0
        row.update(info)
        row["time_s"] = round(elapsed, 1)

        pm_str = f"PM={info['pm_similarity']:.3f}" if info["pm_similarity"] else "PM=✗"
        dg_str = f"DG={info['dg_similarity']:.3f}" if info["dg_similarity"] else "DG=✗"
        best_str = f"Best={info['best_similarity']:.3f} ({info['best_method']})" if info["best_similarity"] else "No match"
        print(f"  {pm_str} | {dg_str} | {best_str} [{elapsed:.1f}s]")

        if info["best_poscar"]: print_structure(info)

        row["atoms"] = poscar_to_atoms(info.get("best_poscar"))
        pm_atoms = poscar_to_atoms(info.get("pm_poscar"))
        dg_atoms = poscar_to_atoms(info.get("dg_poscar"))
        row["pm_atoms_dict"] = pm_atoms.to_dict() if pm_atoms else None
        row["dg_atoms_dict"] = dg_atoms.to_dict() if dg_atoms else None
        row["best_atoms_dict"] = row["atoms"].to_dict() if row["atoms"] else None

        rruff_cell = parse_rruff_cell_params(entry.get("##CELL PARAMETERS",""))
        matched_atoms, cell_label, matched_params = best_cell_match(row["atoms"], rruff_cell)

        if matched_params is not None:
            for k in ["a","b","c","alpha","beta","gamma","volume"]: row[f"pred_{k}"] = matched_params[k]
            row["cell_type"] = matched_params["cell_type"]
            if cell_label != "original": print(f"  → Using {cell_label} cell for lattice comparison")
            print_cell_comparison(rruff_cell, matched_params, cell_label)
        elif row["atoms"] is not None:
            lat = row["atoms"].lattice
            for k,v in [("a",lat.a),("b",lat.b),("c",lat.c),("alpha",lat.alpha),("beta",lat.beta),("gamma",lat.gamma),("volume",lat.volume)]:
                row[f"pred_{k}"] = v
            row["cell_type"] = "original"
            if rruff_cell:
                fb = {"a":lat.a,"b":lat.b,"c":lat.c,"alpha":lat.alpha,"beta":lat.beta,"gamma":lat.gamma,"volume":lat.volume}
                print_cell_comparison(rruff_cell, fb, "original")

    except requests.exceptions.HTTPError as e:
        row["error"] = f"HTTP {e.response.status_code}: {e}"
        row["time_s"] = None; row["atoms"] = None
        print(f"  ERROR: {row['error']}")
        if e.response.status_code == 429:
            print(f"  ⏳ Extended cooldown: {cfg['max_backoff']:.0f}s")
            time.sleep(cfg["max_backoff"])
    except Exception as e:
        row["error"] = str(e); row["time_s"] = None; row["atoms"] = None
        print(f"  ERROR: {e}")

    return row


def lattice_comparison(df, cfg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import mean_absolute_error, r2_score

    params = ["a","b","c","alpha","beta","gamma","volume"]
    units = {"a":"Å","b":"Å","c":"Å","alpha":"°","beta":"°","gamma":"°","volume":"ų"}
    mask = pd.Series(True, index=df.index)
    for p in params: mask &= df[f"rruff_{p}"].notna() & df[f"pred_{p}"].notna()
    df_cmp = df[mask].copy(); n = len(df_cmp)
    if n < 2: print(f"\n⚠ Only {n} entries — skipping comparison."); return

    print(f"\n{'═'*60}\nLATTICE PARAMETER COMPARISON (n={n})\n{'═'*60}")
    print(f"{'Parameter':>10s}  {'MAE':>10s}  {'R²':>10s}  {'Unit':>5s}\n{'─'*40}")
    metrics = {}
    for p in params:
        exp,pred = df_cmp[f"rruff_{p}"].values.astype(float), df_cmp[f"pred_{p}"].values.astype(float)
        mae,r2 = mean_absolute_error(exp,pred), (r2_score(exp,pred) if np.std(exp)>1e-10 else float("nan"))
        metrics[p] = {"mae":mae,"r2":r2,"exp":exp,"pred":pred}
        print(f"{p:>10s}  {mae:10.4f}  {r2:10.4f}  {units[p]:>5s}")

    fig, axes = plt.subplots(2,4,figsize=(20,10)); axes=axes.flatten()
    for i,p in enumerate(params):
        ax=axes[i]; exp,pred,mae,r2 = metrics[p]["exp"],metrics[p]["pred"],metrics[p]["mae"],metrics[p]["r2"]
        ax.scatter(exp,pred,c="#3b82f6",s=40,alpha=0.7,edgecolors="white",linewidths=0.5)
        lo,hi = np.concatenate([exp,pred]).min()*0.95, np.concatenate([exp,pred]).max()*1.05
        if lo==hi: lo-=1;hi+=1
        ax.plot([lo,hi],[lo,hi],"--",color="gray",alpha=0.5); ax.set_xlim(lo,hi); ax.set_ylim(lo,hi)
        ax.set_aspect("equal"); ax.set_xlabel(f"RRUFF {p} ({units[p]})"); ax.set_ylabel(f"Pred {p} ({units[p]})")
        ax.set_title(f"{p} MAE={mae:.4f} R²={r2:.4f}",fontweight="bold"); ax.grid(True,alpha=0.2)
    for j in range(len(params),len(axes)): axes[j].set_visible(False)
    fig.suptitle(f"RRUFF vs Predicted (n={n})",fontsize=14,fontweight="bold",y=1.01)
    plt.tight_layout()
    plot_path = cfg.get("output_csv","results.csv").replace(".csv","_lattice_parity.png")
    fig.savefig(plot_path,dpi=150,bbox_inches="tight"); plt.close(fig)
    print(f"\n✓ Parity plots → {plot_path}")


def run_pipeline(cfg):
    headers = {"Authorization":f"Bearer {cfg['api_key']}","accept":"application/json","Content-Type":"application/json"}

    print("Loading RRUFF powder XRD database...")
    d = figshare_data("rruff_powder_xrd")
    print(f"Total RRUFF entries: {len(d)}")

    # De-duplicate
    seen, rruff_unique, skipped = set(), [], 0
    for entry in d:
        if "##IDEAL CHEMISTRY" not in entry: skipped+=1; continue
        formula = _get_formula(entry["##IDEAL CHEMISTRY"])
        if not formula: skipped+=1; continue
        elements = _get_elements(formula)
        if not elements:
            print(f"  ⚠ Skipping {entry.get('##NAMES','?')}: no elements from '{entry['##IDEAL CHEMISTRY']}'")
            skipped+=1; continue
        name = entry["##NAMES"]
        if name not in seen: seen.add(name); rruff_unique.append(entry)
    print(f"Unique minerals with valid chemistry: {len(rruff_unique)} (skipped {skipped})")

    # ── Filter by RRUFF cell parameter size ──
    if cfg.get("max_abc") is not None:
        max_val = cfg["max_abc"]
        before = len(rruff_unique)
        filtered = []
        for entry in rruff_unique:
            cell = parse_rruff_cell_params(entry.get("##CELL PARAMETERS",""))
            if cell is None:
                filtered.append(entry)  # keep entries without cell params
                continue
            if cell["a"] <= max_val and cell["b"] <= max_val and cell["c"] <= max_val:
                filtered.append(entry)
        rruff_unique = filtered
        print(f"Filtered by RRUFF a,b,c ≤ {max_val} Å: {len(rruff_unique)}/{before}")

    if cfg["max_entries"] is not None:
        rruff_unique = rruff_unique[:cfg["max_entries"]]
        print(f"Processing first {cfg['max_entries']} entries")

    n_total = len(rruff_unique)
    print(f"\n{'═'*60}")
    print(f"Method: {cfg['method']} | Wavelength: {cfg['wavelength']}Å | Refinement: {cfg['run_refinement']} ({cfg['refinement_engine']}) | Delay: {cfg['delay']}s")
    if cfg.get("max_abc"): print(f"Cell filter: RRUFF a,b,c ≤ {cfg['max_abc']} Å")
    print(f"{'═'*60}")

    work_dir = os.path.dirname(os.path.abspath(cfg["output_csv"])) or "."
    os.makedirs(work_dir, exist_ok=True)
    cache_dir = os.path.join(work_dir, cfg["cache_dir"])
    os.makedirs(cache_dir, exist_ok=True)

    if cfg.get("overwrite_cache"):
        for fn in os.listdir(cache_dir):
            if fn.endswith(".json"):
                try:
                    os.remove(os.path.join(cache_dir, fn))
                except Exception:
                    pass

    cached_rows = _load_cached_results(cache_dir) if cfg.get("resume", True) else {}
    if cached_rows:
        print(f"Found {len(cached_rows)} cached entries in {cache_dir}")

    results = []
    t_start = time.time()
    for idx, entry in enumerate(rruff_unique):
        name = entry["##NAMES"]
        formula = _get_formula(entry["##IDEAL CHEMISTRY"])
        elements = _get_elements(formula)
        n_pts = len(entry["x"])
        rid = entry["##RRUFFID"]
        entry_id = _entry_uid(entry)

        elapsed_total = time.time() - t_start
        avg_per = elapsed_total / max(idx, 1)
        eta_min = avg_per * (n_total - idx) / 60
        elapsed_min = elapsed_total / 60

        print(f"\n[{idx+1}/{n_total}] {name} {rid} ({formula}) — {n_pts} pts, elements={elements}  [elapsed: {elapsed_min:.1f}min, ETA: {eta_min:.1f}min]")

        if entry_id in cached_rows:
            row = dict(cached_rows[entry_id])
            row["entry_id"] = entry_id
            row["atoms"] = None
            results.append(row)
            if row.get("error") is None:
                print("  ↺ Loaded from cache")
            else:
                print(f"  ↺ Loaded failed cached result: {row.get('error')}")
            continue

        row = process_entry(entry, headers, cfg)
        row["entry_id"] = entry_id
        results.append(row)
        _checkpoint_row(row, cache_dir)
        if ((idx + 1) % max(cfg.get("write_every", 1), 1) == 0) or (idx == n_total - 1):
            _write_live_outputs(results, cfg, work_dir)
        if idx < n_total - 1: time.sleep(cfg["delay"])

    total_elapsed = time.time() - t_start
    print(f"\n✓ Pipeline complete: {n_total} entries in {total_elapsed/60:.1f} min ({total_elapsed/max(n_total,1):.1f} s/entry avg)")

    # Retry failed
    df = pd.DataFrame(results)
    failed_indices = df[df["error"].notna()].index.tolist()
    if failed_indices:
        retry_delay = cfg["delay"] * 3
        print(f"\n{'═'*60}\nRetrying {len(failed_indices)} failed entries with {retry_delay:.0f}s delay...\n{'═'*60}")
        for df_idx in failed_indices:
            row_data = df.loc[df_idx]; name,formula = row_data["mineral_name"],row_data["formula"]
            entry = next((e for e in rruff_unique if e["##NAMES"]==name), None)
            if entry is None: continue
            print(f"\n  Retrying: {name} ({formula})...")
            new_row = process_entry(entry, headers, cfg)
            new_row["entry_id"] = row_data.get("entry_id") or _entry_uid(entry)
            if new_row.get("error") is None:
                for k,v in new_row.items():
                    if k in df.columns: df.at[df_idx,k] = v
                    else: df[k] = None; df.at[df_idx,k] = v
                print(f"    ✓ Retry succeeded")
            else:
                for k,v in new_row.items():
                    if k in df.columns: df.at[df_idx,k] = v
                    else: df[k] = None; df.at[df_idx,k] = v
                print(f"    ✗ Still failed: {new_row['error']}")
            cache_row = df.loc[df_idx].to_dict()
            cache_row["atoms"] = None
            _checkpoint_row(cache_row, cache_dir)
            _write_live_outputs(df.to_dict("records"), cfg, work_dir)
            time.sleep(retry_delay)

    # Summary
    total = len(df)
    pm_ok = df["pm_success"].sum() if "pm_success" in df.columns else 0
    dg_ok = df["dg_success"].sum() if "dg_success" in df.columns else 0
    any_match = ((df.get("pm_success",False))|(df.get("dg_success",False))).sum()
    errors = df["error"].notna().sum()

    print(f"\n{'═'*60}\nRESULTS SUMMARY\n{'═'*60}")
    print(f"Total minerals processed:  {total}")
    print(f"Pattern matching success:  {pm_ok}/{total} ({100*pm_ok/total:.1f}%)")
    print(f"DiffractGPT success:       {dg_ok}/{total} ({100*dg_ok/total:.1f}%)")
    print(f"Any method matched:        {any_match}/{total} ({100*any_match/total:.1f}%)")
    print(f"Errors remaining:          {errors}")

    if pm_ok > 0:
        s = df.loc[df["pm_success"],"pm_similarity"]
        print(f"\nPM similarity — mean:{s.mean():.3f} med:{s.median():.3f} min:{s.min():.3f} max:{s.max():.3f}")
    if dg_ok > 0:
        s = df.loc[df["dg_success"],"dg_similarity"]
        print(f"DG similarity — mean:{s.mean():.3f} med:{s.median():.3f} min:{s.min():.3f} max:{s.max():.3f}")

    matched = df[df["best_similarity"].notna()].sort_values("best_similarity",ascending=False)
    if len(matched) > 0:
        print(f"\n{'─'*60}\nTOP 5 BEST MATCHES:")
        for _,r in matched.head(5).iterrows():
            print(f"  {r['mineral_name']:20s} {r['formula']:15s} → sim={r['best_similarity']:.4f} ({r['best_method']}) ID={r.get('pm_jid','N/A')}")
        print("\nBOTTOM 5 WORST MATCHES:")
        for _,r in matched.tail(5).iterrows():
            print(f"  {r['mineral_name']:20s} {r['formula']:15s} → sim={r['best_similarity']:.4f} ({r['best_method']}) ID={r.get('pm_jid','N/A')}")

    unmatched = df[df["best_similarity"].isna() & df["error"].isna()]
    if len(unmatched)>0:
        print(f"\nUNMATCHED ({len(unmatched)}):")
        for _,r in unmatched.iterrows(): print(f"  {r['mineral_name']:20s} {r['formula']:15s}")
    if errors>0:
        print(f"\nERRORS ({errors}):")
        for _,r in df[df["error"].notna()].iterrows(): print(f"  {r['mineral_name']:20s} → {r['error'][:80]}")

    _, consolidated, _ = _save_final_results(df, cfg)
    return df, consolidated


# ═══════════════════════════════════════════════════════════════════════════
# def _in_notebook():
#     try:
#         from IPython import get_ipython
#         return get_ipython().__class__.__name__ in ("ZMQInteractiveShell","TerminalInteractiveShell")
#     except Exception: return False
def _in_notebook():
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
    except Exception:
        pass
    # Fallback: Jupyter/Colab kernels pass -f
    return "-f" in sys.argv
def parse_args():
    if _in_notebook(): return dict(DEFAULTS)
    p = argparse.ArgumentParser(description="RRUFF XRD → AtomGPT.org", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--base-url", default=DEFAULTS["base_url"])
    p.add_argument("--api-key", default=DEFAULTS["api_key"])
    p.add_argument("--method", default=DEFAULTS["method"], choices=["pattern_matching","diffractgpt","both"])
    p.add_argument("--wavelength", type=float, default=DEFAULTS["wavelength"])
    p.add_argument("--max-entries", type=int, default=DEFAULTS["max_entries"], help="0 = all")
    p.add_argument("--max-abc", type=float, default=None, help="Filter: RRUFF a,b,c ≤ this (Å)")
    p.add_argument("--no-element-fallback", action="store_true")
    p.add_argument("--run-refinement", action="store_true")
    p.add_argument("--refinement-engine", default=DEFAULTS["refinement_engine"], choices=["gsas2","bmgn","auto"])
    p.add_argument("--preferred-orientation", action="store_true")
    p.add_argument("--run-alignn", action="store_true")
    p.add_argument("--run-slakonet", action="store_true")
    p.add_argument("--delay", type=float, default=DEFAULTS["delay"])
    p.add_argument("--max-retries", type=int, default=DEFAULTS["max_retries"])
    p.add_argument("--initial-backoff", type=float, default=DEFAULTS["initial_backoff"])
    p.add_argument("--max-backoff", type=float, default=DEFAULTS["max_backoff"])
    p.add_argument("--output-csv", default=DEFAULTS["output_csv"])
    p.add_argument("--output-poscars", default=DEFAULTS["output_poscars"])
    p.add_argument("--cache-dir", default=DEFAULTS["cache_dir"])
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--overwrite-cache", action="store_true")
    p.add_argument("--write-every", type=int, default=DEFAULTS["write_every"])
    args = p.parse_args()
    return {
        "base_url":args.base_url,"api_key":args.api_key,"method":args.method,
        "wavelength":args.wavelength,
        "max_entries":args.max_entries if args.max_entries>0 else None,
        "max_abc":args.max_abc,
        "fallback_to_elements":not args.no_element_fallback,
        "run_refinement":args.run_refinement,"refinement_engine":args.refinement_engine,
        "preferred_orientation":args.preferred_orientation,
        "run_alignn":args.run_alignn,"run_slakonet":args.run_slakonet,
        "delay":args.delay,"max_retries":args.max_retries,
        "initial_backoff":args.initial_backoff,
        "backoff_multiplier":DEFAULTS["backoff_multiplier"],
        "max_backoff":args.max_backoff,
        "output_csv":args.output_csv,"output_poscars":args.output_poscars,
        "cache_dir":args.cache_dir,"resume":not args.no_resume,
        "overwrite_cache":args.overwrite_cache,
        "write_every":max(1, args.write_every),
    }

if __name__ == "__main__":
    cfg = parse_args()
    run_pipeline(cfg)
