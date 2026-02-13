import argparse
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Crippen, QED
from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles
from rdkit.Chem.rdMolDescriptors import CalcTPSA

try:
    from rdkit.Chem import RDConfig
    import os
    import sys
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    import sascorer
except Exception:
    sascorer = None


def safe_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Compute Bemis–Murcko scaffolds and properties.")
    parser.add_argument("--input", required=True, help="Input CSV path")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--smiles_col", default="smiles", help="SMILES column name")
    parser.add_argument("--scaffold_col", default="scaffold_smiles", help="Scaffold column name")
    parser.add_argument("--add_props", nargs="+", default=["logp", "sas", "tpsa", "qed"],
                        help="Properties to add: logp sas tpsa qed")
    parser.add_argument("--drop_invalid", action="store_true", help="Drop rows with invalid SMILES")
    parser.add_argument("--chunksize", type=int, default=50000, help="Rows per chunk")
    args = parser.parse_args()

    smiles_col = args.smiles_col.lower()
    scaffold_col = args.scaffold_col.lower()
    props = [p.lower() for p in args.add_props]

    first_chunk = True
    total_rows = 0

    for chunk_idx, df in enumerate(pd.read_csv(args.input, chunksize=args.chunksize)):
        df.columns = df.columns.str.lower()

        if smiles_col not in df.columns:
            raise ValueError(f"SMILES column '{smiles_col}' not found")

        mols = df[smiles_col].astype(str).apply(safe_mol)

        if args.drop_invalid:
            valid_mask = mols.notna()
            df = df[valid_mask].reset_index(drop=True)
            mols = mols[valid_mask].reset_index(drop=True)

        df[scaffold_col] = mols.apply(lambda m: MurckoScaffoldSmiles(mol=m) if m is not None else "")

        if "logp" in props:
            df["logp"] = mols.apply(lambda m: round(Crippen.MolLogP(m), 4) if m is not None else float("nan"))
        if "tpsa" in props:
            df["tpsa"] = mols.apply(lambda m: round(CalcTPSA(m), 4) if m is not None else float("nan"))
        if "qed" in props:
            df["qed"] = mols.apply(lambda m: round(QED.qed(m), 4) if m is not None else float("nan"))
        if "sas" in props:
            if sascorer is None:
                raise RuntimeError("sascorer not available. Ensure RDKit contrib SA_Score is installed.")
            df["sas"] = mols.apply(lambda m: round(sascorer.calculateScore(m), 4) if m is not None else float("nan"))

        df.to_csv(args.output, index=False, mode="w" if first_chunk else "a", header=first_chunk)
        total_rows += len(df)
        print(f"Processed chunk {chunk_idx + 1}, total rows written: {total_rows}")
        first_chunk = False


if __name__ == "__main__":
    main()
