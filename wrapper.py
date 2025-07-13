#!/usr/bin/env python3
"""
TF-motif Search & Annotate (pure Python)

Отображение глобального прогресса всех этапов работы.
"""

import argparse
import time
import tracemalloc
from collections import defaultdict
import os
import platform
import sys

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio import pairwise2
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(
        prog="wrapper.py",
        description="Поиск TF-мотивов и аннотация результатов в CSV"
    )
    p.add_argument(
        "--genbank", required=True,
        help="путь к файлу GenBank с аннотациями и последовательностью"
    )
    p.add_argument(
        "--fasta", required=True,
        help="путь к FASTA-файлу с ридами/шаблонами"
    )
    p.add_argument(
        "--out", required=True,
        help="путь для сохранения выходного CSV (разделитель — точка-с-запятой)"
    )
    p.add_argument(
        "--k", type=int, default=8,
        help="длина seed k-меров (по умолчанию %(default)s)"
    )
    p.add_argument(
        "--band", type=int, default=50,
        help="ширина полосы для локального выравнивания (по умолчанию %(default)s)"
    )
    return p.parse_args()


def main():
    args = parse_args()

    # 1) Читаем GenBank и собираем CDS-аннотации
    gb = SeqIO.read(args.genbank, "genbank")
    genome = str(gb.seq)
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

    # 2) Загружаем FASTA-риды
    reads = list(SeqIO.parse(args.fasta, "fasta"))
    n_reads = len(reads)

    # Веса этапов (количество «шагов» в каждом)
    w_parse      = 1
    w_index      = max(len(genome) - args.k + 1, 1)
    w_align      = n_reads
    w_annotate   = n_reads
    w_write      = 1
    w_metrics    = 1

    total_steps = w_parse + w_index + w_align + w_annotate + w_write + w_metrics

    # Запускаем прогресс-бар
    pbar = tqdm(total=total_steps, desc="Overall progress", unit="step")

    # Параметры для замеров
    start_time = time.time()
    tracemalloc.start()

    # Шаг 1: парсинг GenBank
    pbar.update(w_parse)

    # Шаг 2: строим индекс k-меров
    kmer_index = defaultdict(list)
    G = len(genome)
    for i in range(G - args.k + 1):
        kmer_index[genome[i : i + args.k]].append(i)
        pbar.update(1)

    # Шаг 3: выравнивание (seed-and-extend на каждом риде)
    alignments = []
    for rec in reads:
        read = str(rec.seq)
        best_score = float("-inf")
        best_pos = 0
        best_aln = ""
        for j in range(len(read) - args.k + 1):
            seed = read[j : j + args.k]
            if seed not in kmer_index:
                continue
            for pos in kmer_index[seed]:
                start = max(0, pos - j - args.band)
                end   = min(G, pos - j + len(read) + args.band)
                ref_seg = genome[start:end]
                alns = pairwise2.align.localms(
                    ref_seg, read,
                    2, -1,    # match, mismatch
                    -2, -0.5, # gap open, extend
                    one_alignment_only=True
                )
                if alns:
                    _, seqB, score, begin, _ = alns[0]
                    if score > best_score:
                        best_score = score
                        best_pos   = start + begin
                        best_aln   = seqB
        if best_score == float("-inf"):
            alignments.append((0, "+", "", 0.0))
        else:
            alignments.append((best_pos, "+", best_aln, best_score))
        pbar.update(1)

    # Шаг 4: аннотация результатов
    # собираем DataFrame и сортируем по pos
    df_aln = pd.DataFrame(
        alignments, columns=["pos","strand","seq","score"]
    ).sort_values("pos")
    # merge_asof
    df_merged = pd.merge_asof(
        df_aln, df_genes,
        left_on="pos", right_on="gene_start",
        direction="backward"
    )
    df_merged["Dist to ATG"] = df_merged["pos"] - df_merged["ATG_start"]
    pbar.update(w_annotate)

    # Шаг 5: запись CSV
    df_out = df_merged[[
        "pos","strand","seq","Product","Locus","Dist to ATG","score"
    ]].rename(columns={
        "pos":"Position","seq":"Sequence","score":"Score"
    })
    df_out.to_csv(args.out, sep=";", index=False)
    pbar.update(w_write)

    # Шаг 6: сбор и вывод метрик
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    total_time = time.time() - start_time
    throughput = n_reads / total_time if total_time>0 else 0.0
    identities = [
        (score / (2*len(seq))) if seq else 0.0
        for _,_,seq,score in alignments
    ]
    mean_identity = sum(identities)/len(identities) if identities else 0.0

    pbar.update(w_metrics)
    pbar.close()

    print(f"Throughput (reads/sec): {throughput:.2f}")
    print(f"Mean identity:          {mean_identity:.2%}")
    print(f"Total time (s):         {total_time:.3f}")
    print(f"Peak memory (bytes):    {peak}")


if __name__ == "__main__":
    main()
