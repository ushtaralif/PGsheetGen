import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

Record = Dict[str, Any]
class Sampling:
    def __init__(self):
        print()

    def _detect_format(self, path: str) -> str:
        ext = os.path.splitext(path.lower())[1]
        if ext == ".jsonl":
            return "jsonl"
        if ext == ".json":
            return "json"
        raise ValueError("Input file must have extension .json or .jsonl")


    def _read_records(self,path: str) -> List[Record]:
        fmt = self._detect_format(path)
        if fmt == "json":
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                return obj
            if isinstance(obj, dict):
                return [obj]
            raise ValueError(f"Unsupported JSON top-level type: {type(obj)}")
        else:
            out: List[Record] = []
            with open(path, "r", encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {lineno}: {e}") from e
            return out


    def _edge_type(self,rec: Record) -> str:
        try:
            e = rec.get("Edge", {})
            t = e.get("type", None)
            if isinstance(t, str) and t.strip():
                return t.strip()
            return "__MISSING_EDGE_TYPE__"
        except Exception:
            return "__MALFORMED_RECORD__"


    def _pct(self,x: int, total: int) -> str:
        if total <= 0:
            return "0.00%"
        return f"{(100.0 * x / total):.2f}%"


    def compute_type_counts(self,records: List[Record]) -> Dict[str, int]:
        counts = defaultdict(int)
        for rec in records:
            counts[self._edge_type(rec)] += 1
        return dict(counts)


    def compute_chunk_type_counts(self,chunks: List[List[Record]]) -> List[Dict[str, int]]:
        out: List[Dict[str, int]] = []
        for chunk in chunks:
            counts = defaultdict(int)
            for rec in chunk:
                counts[self._edge_type(rec)] += 1
            out.append(dict(counts))
        return out


    def print_stats(self,records_used: List[Record], chunks: List[List[Record]], top_k: int = 30) -> None:
        total = len(records_used)
        num_chunks = len(chunks)
        global_counts = self.compute_type_counts(records_used)
        types_sorted = sorted(global_counts.items(), key=lambda x: (-x[1], x[0]))
        n_types = len(types_sorted)

        print("=" * 70)
        print("Global Stratified Partition Statistics")
        print()
        print("=" * 70)
        print(f"Total records              : {total}")
        print(f"Number of edge types       : {n_types}")
        print(f"Number of chunks           : {num_chunks}")
        print("Chunk sizes                : " + ", ".join(str(len(c)) for c in chunks))
        print()

        print(f"Global edge-type distribution (top {min(top_k, n_types)}):")
        head = types_sorted[:top_k]
        for t, c in head:
            print(f"  {t:30s}  {c:10d}  {self._pct(c, total):>8s}")
        if n_types > top_k:
            tail_count = sum(c for _, c in types_sorted[top_k:])
            print(f"  {'[TAIL_TYPES]':30s}  {tail_count:10d}  {self._pct(tail_count, total):>8s}")
        print()

        chunk_type_counts = self.compute_chunk_type_counts(chunks)
        all_types = set(global_counts.keys())

        print("Per-chunk missing edge types:")
        for i, cc in enumerate(chunk_type_counts):
            present = set(cc.keys())
            missing = all_types - present
            print(f"  Chunk {i:2d}: size={len(chunks[i]):8d}  missing_types={len(missing):4d}")
        print()

        print(f"Per-chunk counts for top {min(top_k, n_types)} edge types:")
        top_types = [t for t, _ in head]
        header = "Type".ljust(32) + " | " + " ".join([f"C{i:02d}".rjust(7) for i in range(num_chunks)])
        print(header)
        print("-" * len(header))
        for t in top_types:
            row = t.ljust(32) + " | "
            for i in range(num_chunks):
                row += f"{chunk_type_counts[i].get(t, 0):7d}"
            print(row)
        print()

        total_in_chunks = sum(len(c) for c in chunks)
        print("Sanity checks:")
        print(f"  Total records in chunks   : {total_in_chunks}  (should equal {total})")
        if total_in_chunks != total:
            print("  WARNING: Totals do not match; check logic.")

        ok = True
        for t, c in global_counts.items():
            s = sum(chunk_type_counts[i].get(t, 0) for i in range(num_chunks))
            if s != c:
                ok = False
                print(f"  WARNING: type={t} global={c} sum_chunks={s}")
        print(f"  Per-type totals preserved : {'YES' if ok else 'NO'}")
        print("=" * 70)
        print()

    def _largest_remainder_alloc(self,proportions: Dict[str, float], total: int) -> Dict[str, int]:
        raw = {k: proportions[k] * total for k in proportions}
        alloc = {k: int(raw[k]) for k in proportions}
        remaining = total - sum(alloc.values())

        remainders = sorted(((raw[k] - alloc[k], k) for k in proportions), reverse=True)
        i = 0
        while remaining > 0:
            _, k = remainders[i % len(remainders)]
            alloc[k] += 1
            remaining -= 1
            i += 1
        return alloc


    def alloc_with_min_per_type(self,
        proportions: Dict[str, float],
        chunk_size: int,
        types: List[str],
        min_per_type: int = 1,
    ) -> Dict[str, int]:

        if chunk_size < min_per_type * len(types):
            raise ValueError(
                f"chunk_size={chunk_size} is too small to guarantee min_per_type={min_per_type} "
                f"for all {len(types)} edge types."
            )

        alloc = self._largest_remainder_alloc(proportions, chunk_size)

        needed = 0
        for t in types:
            if alloc.get(t, 0) < min_per_type:
                needed += (min_per_type - alloc.get(t, 0))
                alloc[t] = min_per_type

        if needed == 0:
            return alloc

        expected = {t: proportions[t] * chunk_size for t in types}

        candidates = sorted(
            types,
            key=lambda t: ((alloc[t] - expected[t]), alloc[t]),
            reverse=True
        )

        i = 0
        while needed > 0:
            if i >= len(candidates):
                candidates = sorted(types, key=lambda t: alloc[t], reverse=True)
                i = 0

            t = candidates[i]
            if alloc[t] > min_per_type:
                alloc[t] -= 1
                needed -= 1
            i += 1

        s = sum(alloc[t] for t in types)
        if s != chunk_size:
            raise RuntimeError(f"Internal error: allocation sums to {s}, expected {chunk_size}")

        return alloc


    def build_stratified_bins_from_file(self,
        input_path: str,
        num_chunks: int,
        chunk_size: int,
        seed: int = 42,
        allow_replacement: bool = False,
    ) -> Tuple[List[List[Record]], List[Record]]:

        if num_chunks <= 0:
            raise ValueError("num_chunks must be >= 1")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be >= 1")

        rng = random.Random(seed)
        records = self._read_records(input_path)
        if not records:
            return ([[] for _ in range(num_chunks)], [])

        pools: Dict[str, List[Record]] = defaultdict(list)
        for rec in records:
            pools[self._edge_type(rec)].append(rec)

        types = sorted(pools.keys())
        counts = {t: len(pools[t]) for t in types}
        N = sum(counts.values())
        target_total = num_chunks * chunk_size

        if target_total > N and not allow_replacement:
            raise ValueError(
                f"Requested num_chunks*chunk_size={target_total} exceeds total records={N}. "
                f"Set allow_replacement=True or reduce num_chunks/chunk_size."
            )

        for t in types:
            rng.shuffle(pools[t])

        proportions = {t: counts[t] / N for t in types}

        per_chunk_alloc = self.alloc_with_min_per_type(
            proportions=proportions,
            chunk_size=chunk_size,
            types=types,
            min_per_type=1,
        )

        ptr = {t: 0 for t in types}

        def draw_one(t: str) -> Record:
            if ptr[t] < len(pools[t]):
                rec = pools[t][ptr[t]]
                ptr[t] += 1
                return rec
            if allow_replacement and pools[t]:
                return rng.choice(pools[t])
            raise RuntimeError(f"Pool exhausted for type={t}. Consider allow_replacement=True.")

        chunks: List[List[Record]] = []
        records_used: List[Record] = []

        for _c in range(num_chunks):
            chunk: List[Record] = []

            for t in types:
                k = per_chunk_alloc.get(t, 0)
                for _ in range(k):
                    r = draw_one(t)
                    chunk.append(r)
                    records_used.append(r)

            while len(chunk) < chunk_size:
                candidates = []
                weights = []
                for t in types:
                    avail = (len(pools[t]) - ptr[t]) if not allow_replacement else len(pools[t])
                    if avail > 0:
                        candidates.append(t)
                        weights.append(avail)
                if not candidates:
                    raise RuntimeError("No candidates available to fill chunk.")
                t_sel = rng.choices(candidates, weights=weights, k=1)[0]
                r = draw_one(t_sel)
                chunk.append(r)
                records_used.append(r)

            rng.shuffle(chunk)
            chunks.append(chunk)

        if len(chunks) != num_chunks:
            raise RuntimeError("Internal error: incorrect number of chunks produced.")
        for ch in chunks:
            if len(ch) != chunk_size:
                raise RuntimeError("Internal error: chunk_size constraint violated.")
        if len(records_used) != target_total:
            raise RuntimeError("Internal error: records_used size mismatch.")

        return chunks, records_used
