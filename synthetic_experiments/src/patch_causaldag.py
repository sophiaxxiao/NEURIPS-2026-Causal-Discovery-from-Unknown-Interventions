"""
patch_causaldag.py
------------------
Applies bug fixes to the installed causaldag package.
Run once after installing causaldag in any environment:

    python patch_causaldag.py
"""
import causaldag, os

root = os.path.dirname(causaldag.__file__)

patches = {
    # 1. Replace numpy.linalg.inv with pinv in the Gaussian invariance test
    #    to handle near-singular gram matrices.
    os.path.join(root, "utils", "invariance_tests", "gauss_invariance.py"): [
        (
            "from numpy.linalg import inv\n",
            "from numpy.linalg import pinv as inv\n",
        ),
        (
            "        coefs1 = np.linalg.inv(gram1) @ obs_samples[:, cond_ix].T @ obs_samples[:, i]\n"
            "        coefs2 = np.linalg.inv(gram2) @ iv_samples[:, cond_ix].T @ iv_samples[:, i]\n",
            "        coefs1 = np.linalg.pinv(gram1) @ obs_samples[:, cond_ix].T @ obs_samples[:, i]\n"
            "        coefs2 = np.linalg.pinv(gram2) @ iv_samples[:, cond_ix].T @ iv_samples[:, i]\n",
        ),
    ],
    # 2. Fix _is_icovered in igsp: setting_num was wrongly passed as cond_set.
    os.path.join(root, "structure_learning", "dag", "gsp.py"): [
        (
            "        if setting_num is not None and not invariance_tester.is_invariant(j, 0, setting_num):\n",
            "        if setting_num is not None and not invariance_tester.is_invariant(j, setting_num):\n",
        ),
    ],
}

for path, replacements in patches.items():
    with open(path, "r") as f:
        src = f.read()
    original = src
    for old, new in replacements:
        if new in src:
            print(f"  [already applied] {os.path.basename(path)}: {repr(old[:40])}")
            continue
        if old not in src:
            print(f"  [NOT FOUND — version mismatch?] {os.path.basename(path)}: {repr(old[:40])}")
            continue
        src = src.replace(old, new, 1)
        print(f"  [patched] {os.path.basename(path)}: {repr(old[:40])}")
    if src != original:
        with open(path, "w") as f:
            f.write(src)

print("Done.")
