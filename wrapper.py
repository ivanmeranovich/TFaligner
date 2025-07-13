#!/usr/bin/env python3
"""
TF-motif Search & Annotate (pure Python)

Поиск TF-мотивов в референсном геноме с аннотацией ближайших CDS.
"""

import argparse
import time
import tracemalloc
from collections import defaultdict

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2


def parse_args():
    p = argparse.ArgumentParser(
        prog="wrapper.py",
        description="Поиск TF-мотивов и аннотация результатов в CSV"
    )
    p.add_argument(
        "--genbank",
        required=True,
        help="путь к файлу GenBank с аннотациями и последовательностью генома"
    )
    p.add_argument(
        "--fasta",
        required=True,
        help="путь к FASTA-файлу с ридами/мотивами для поиска"
    )
    p.add_argument(
        "--out",
        required=True,
        help="путь для сохранения выходного CSV (точка с запятой как разделитель)"
    )
    p.add_argument(
        "--k",
        type=int,
        default=8,
        help="длина seed k-меров (по умолчанию %(default)s)"
    )
    p.add_argument(
        "--band",
        type=int,
        default=50,
        help="ширина полосы для локального выравнивания (по умолчанию %(default)s)"
    )
    return p.parse_args()


def build_kmer_index(genome: str, k: int) -> dict:
    idx = defaultdict(list)
    for i in range(len(genome) - k + 1):
        idx[genome[i : i + k]].append(i)
    return idx


def align_read(genome: str, index: dict, read: str, k: int, band: int):
    best_score = float("-inf")
    best_pos = 0
    best_strand = "+"
    best_aln = ""
    for j in range(len(read) - k + 1):
        seed = read[j : j + k]
        if seed not in index:
            continue
        for pos in index[seed]:
            start = max(0, pos - j - band)
            end = min(len(genome), pos - j + len(read) + band)
            ref_seg = genome[start:end]
            alns = pairwise2.align.localms(
                ref_seg, read,
                2, -1,    # match, mismatch
                -2, -0.5, # gap open, extend
                one_alignment_only=True
            )
            if not alns:
                continue
            seqA, seqB, score, begin, _ = alns[0]
            if score > best_score:
                best_score = score
                best_pos = start + begin
                best_aln = seqB
    if best_score == float("-inf"):
        return 0, "+", "", 0.0
    return best_pos, best_strand, best_aln, best_score


def align_batch(genome: str, reads: list, k: int, band: int):
    idx = build_kmer_index(genome, k)
    results = []
    for rec in reads:
        pos, strand, seq, score = align_read(genome, idx, str(rec.seq), k, band)
        results.append((pos, strand, seq, score))
    return results


def main():
    args = parse_args()

    # 1. Читаем GenBank
    gb = SeqIO.read(args.genbank, "genbank")
    genome = str(gb.seq)

    # Сбираем аннотации CDS
    genes = []
    for f in gb.features:
        if f.type == "CDS":
            start = int(f.location.start)
            atg = start if genome[start:start+3] == "ATG" else start
            prod = f.qualifiers.get("product", [""])[0]
            locus = f.qualifiers.get("locus_tag", [""])[0]
            genes.append({
                "gene_start": start,
                "ATG_start":  atg,
                "Product":    prod,
                "Locus":      locus
            })
    df_genes = pd.DataFrame(genes).sort_values("gene_start")

    # 2. Загружаем риды
    reads = list(SeqIO.parse(args.fasta, "fasta"))

    # 3. Считаем время и память
    start_time = time.time()
    tracemalloc.start()

    alns = align_batch(genome, reads, args.k, args.band)

    current, peak = tracemalloc.get_traced_memory()
    total_time = time.time() - start_time
    tracemalloc.stop()

    # 4. Рассчитываем метрики
    throughput = len(reads) / total_time if total_time > 0 else 0.0
    identities = [
        (score / (2 * len(seq))) if seq else 0.0
        for _, _, seq, score in alns
    ]
    mean_identity = sum(identities) / len(identities) if identities else 0.0

    # 5. Собираем DataFrame выравниваний
    df_aln = pd.DataFrame(
        alns, columns=["pos","strand","seq","score"]
    ).sort_values("pos")

    # 6. Аннотируем merge_asof
    df_out = pd.merge_asof(
        df_aln, df_genes,
        left_on="pos", right_on="gene_start",
        direction="backward"
    )
    df_out["Dist to ATG"] = df_out["pos"] - df_out["ATG_start"]

    # 7. Сохраняем CSV
    df_final = df_out[[
        "pos","strand","seq","Product","Locus","Dist to ATG","score"
    ]].rename(columns={
        "pos":   "Position",
        "seq":   "Sequence",
        "score": "Score"
    })
    df_final.to_csv(args.out, sep=";", index=False)

    # 8. Печатаем метрики
    print(f"Throughput (reads/sec): {throughput:.2f}")
    print(f"Mean identity:          {mean_identity:.2%}")
    print(f"Total time (s):         {total_time:.3f}")
    print(f"Peak memory (bytes):    {peak}")


if __name__ == "__main__":
    main()
