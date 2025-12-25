from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# =========================
# 입력/출력 설정
# =========================
INPUT_REL_DIRS = [
    Path("org/out_s3_v24/Rx_El6.6Deg_Weight/float"),
    Path("org/out_s3_v24/Rx_Taylor35dB_Weight/float"),
]
INPUT_FILES = [
    "inc_u_+0.00000_ch_float.csv",
    "inc_u_+0.12000_ch_float.csv",
    "inc_u_-0.12000_ch_float.csv",
]

OUTPUT_DIR = Path("org/dataFiles_hdlc, sfpdp 데이터 생성_20251222/generated")

# 템플릿(헤더) 파일
DEFAULT_TEMPLATE = Path("qsfp.csv")
DEFAULT_HEADER_LINES = 9  # 네가 말한 “대략 9줄”

# =========================
# 유틸: CSV 읽기
# =========================
def read_iq_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """CSV에서 I,Q 1296개씩 읽기 (헤더/구분자 유연)"""
    df = pd.read_csv(path, header=None, engine="python")
    num = df.apply(pd.to_numeric, errors="coerce")

    if num.shape[1] >= 2:
        counts = num.notna().sum(axis=0).to_numpy()
        c0, c1 = np.argsort(-counts)[:2]
        i = num.iloc[:, int(c0)].dropna().to_numpy(dtype=np.float64)
        q = num.iloc[:, int(c1)].dropna().to_numpy(dtype=np.float64)
        n = min(len(i), len(q))
        i, q = i[:n], q[:n]
        if len(i) != 1296:
            raise ValueError(f"{path.name}: IQ pairs should be 1296, got {len(i)}")
        return i, q

    flat = num.iloc[:, 0].dropna().to_numpy(dtype=np.float64)
    if len(flat) != 2592:
        raise ValueError(f"{path.name}: expected 2592 (I,Q alternating), got {len(flat)}")
    i = flat[0::2]
    q = flat[1::2]
    if len(i) != 1296:
        raise ValueError(f"{path.name}: IQ pairs should be 1296, got {len(i)}")
    return i, q

# =========================
# Q1.15 + 64bit 패킹
# =========================
def float_to_q15(x: np.ndarray) -> np.ndarray:
    """float -> Q1.15 int16"""
    x = np.asarray(x, dtype=np.float64)
    x = np.clip(x, -1.0, (32767.0 / 32768.0))
    y = np.rint(x * 32768.0)
    y = np.clip(y, -32768, 32767)
    return y.astype(np.int16)

def pack_iqiq_pair(i0, q0, i1, q1) -> np.ndarray:
    """(I0,Q0,I1,Q1) -> uint64 (18컬럼용)"""
    i0u = (i0.astype(np.int32) & 0xFFFF).astype(np.uint64)
    q0u = (q0.astype(np.int32) & 0xFFFF).astype(np.uint64)
    i1u = (i1.astype(np.int32) & 0xFFFF).astype(np.uint64)
    q1u = (q1.astype(np.int32) & 0xFFFF).astype(np.uint64)
    return (i0u << 48) | (q0u << 32) | (i1u << 16) | q1u

def pack_iqiq_dup(i, q) -> np.ndarray:
    """
    (I Q I Q) 같은 샘플을 2번 넣는 64bit (36컬럼용)
    word = I<<48 | Q<<32 | I<<16 | Q
    """
    iu = (i.astype(np.int32) & 0xFFFF).astype(np.uint64)
    qu = (q.astype(np.int32) & 0xFFFF).astype(np.uint64)
    return (iu << 48) | (qu << 32) | (iu << 16) | qu

def to_hex16(u64arr: np.ndarray) -> list[str]:
    return [f"{int(v):016X}" for v in u64arr]

# =========================
# 템플릿 헤더 읽기
# =========================
def read_template_header(template_path: Path, header_lines: int) -> tuple[list[str], int]:
    """
    템플릿에서 상단 header_lines 줄을 그대로 가져오고,
    첫 줄의 컬럼 개수를 기준으로 출력 컬럼(18 or 36)을 결정.
    """
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    lines = template_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < header_lines:
        raise ValueError(f"Template has only {len(lines)} lines, but header_lines={header_lines}")

    header = lines[:header_lines]
    col_count = len(header[0].split(","))
    if col_count not in (18, 36):
        # 그래도 일단 템플릿 기준으로 맞추게끔 경고만
        print(f"[WARN] Template column count = {col_count} (expected 18 or 36). We'll follow template anyway.")
    return header, col_count

# =========================
# 스크린샷 배치 로직
# =========================
def build_rows_36x36(i15: np.ndarray, q15: np.ndarray) -> list[list[tuple[int,int]]]:
    """
    입력 1296개를 36개씩 끊어서 36줄 만들고,
    각 줄을 좌우 반전(우->좌)해서 스크린샷처럼 만든다.
    rows[0] = 가장 아래 줄(physical row 1)
    rows[35] = 가장 위 줄(physical row 36)
    """
    rows = []
    for r in range(36):
        s = r * 36
        e = s + 36
        row = list(zip(i15[s:e], q15[s:e]))[::-1]
        rows.append(row)
    return rows

def rows_to_bfm(rows: list[list[tuple[int,int]]], bfm: int) -> list[list[tuple[int,int]]]:
    if bfm == 1:
        return rows[0:12]
    if bfm == 2:
        return rows[12:24]
    if bfm == 3:
        return rows[24:36]
    raise ValueError("bfm must be 1/2/3")

def bfm_rows_to_grid(bfm_rows: list[list[tuple[int,int]]], template_cols: int) -> list[list[str]]:
    """
    템플릿 컬럼 수에 맞춰 출력 body 생성

     변경 핵심:
    - template_cols == 36 인 경우:
        12줄(각 줄 36 complex)을
        (1줄+2줄), (3줄+4줄), ... 로 묶어서 6줄로 만들고,
        각 칸(64bit)에 (rowA의 I,Q) + (rowB의 I,Q)를 채운다.
        => 6행 x 36열

    - template_cols == 18 인 경우:
        기존처럼 한 줄 안에서 2샘플씩 묶어 18워드로 출력(12행 x 18열)
    """
    out = []

    # 36컬럼 템플릿에서 12줄 -> 6줄로 “줄 병합”
    if template_cols == 36:
        if len(bfm_rows) != 12:
            raise ValueError(f"Expected 12 rows for BFM body, got {len(bfm_rows)}")

        for k in range(0, 12, 2):
            rowA = bfm_rows[k]     # 예: 물리 row1
            rowB = bfm_rows[k + 1] # 예: 물리 row2

            if len(rowA) != 36 or len(rowB) != 36:
                raise ValueError("Each BFM row must have 36 complex samples")

            i0 = np.array([p[0] for p in rowA], dtype=np.int16)
            q0 = np.array([p[1] for p in rowA], dtype=np.int16)
            i1 = np.array([p[0] for p in rowB], dtype=np.int16)
            q1 = np.array([p[1] for p in rowB], dtype=np.int16)

            # (rowA I,Q) + (rowB I,Q) 로 64bit 채우기
            w = pack_iqiq_pair(i0, q0, i1, q1)  # 길이 36짜리 uint64
            out.append(to_hex16(w))             # 36개 HEX

        return out  # 6행 x 36열

    # ----------------------------
    # 18컬럼(기존 로직 유지)
    # ----------------------------
    for row in bfm_rows:
        i = np.array([p[0] for p in row], dtype=np.int16)
        q = np.array([p[1] for p in row], dtype=np.int16)

        if template_cols == 18:
            w = pack_iqiq_pair(i[0::2], q[0::2], i[1::2], q[1::2])
            out.append(to_hex16(w))
        else:
            # 예외 템플릿이면 일단 기존 dup 방식
            w = pack_iqiq_dup(i, q)
            out.append(to_hex16(w))

    return out

# =========================
# 저장: 헤더 + 바디
# =========================
def save_with_header(out_path: Path, header_lines: list[str], grid: list[list[str]]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for line in header_lines:
            f.write(line.rstrip("\n") + "\n")
        for row in grid:
            f.write(",".join(row) + "\n")

# =========================
# 실행
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE), help="qsfp.csv template path")
    ap.add_argument("--header_lines", type=int, default=DEFAULT_HEADER_LINES, help="number of header lines to copy")
    args = ap.parse_args()

    root = Path(".").resolve()
    template_path = (root / args.template).resolve()

    header_block, template_cols = read_template_header(template_path, args.header_lines)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    inputs = []
    for d in INPUT_REL_DIRS:
        for fn in INPUT_FILES:
            p = root / d / fn
            if not p.exists():
                raise FileNotFoundError(f"Missing input: {p}")
            inputs.append(p)

    print(f"[INFO] Found inputs: {len(inputs)} (expected 6)")
    print(f"[INFO] Template: {template_path.name} | header_lines={args.header_lines} | cols={template_cols}")
    total_out = 0

    for in_path in inputs:
        rx_name = in_path.parents[1].name
        tag = in_path.stem
        print(f"\n▶ Processing: {rx_name} / {in_path.name}")

        i_f, q_f = read_iq_csv(in_path)
        i15, q15 = float_to_q15(i_f), float_to_q15(q_f)
        print(f"   [Sample] float({i_f[0]}, {q_f[0]}) -> q15({int(i15[0])}, {int(q15[0])})")

        rows = build_rows_36x36(i15, q15)

        for bfm in (1, 2, 3):
            bfm_rows = rows_to_bfm(rows, bfm)
            grid = bfm_rows_to_grid(bfm_rows, template_cols)

            out_name = f"{rx_name}__{tag}__BFM{bfm}__qsfp.csv"
            out_path = root / OUTPUT_DIR / out_name
            save_with_header(out_path, header_block, grid)

            print(f"   [OK] {out_name}  (header={args.header_lines} lines, body=12 rows)")
            total_out += 1

    print(f"\n[DONE] outputs: {total_out} (expected 18)")
    print(f"[PATH]  { (root / OUTPUT_DIR).as_posix() }")

if __name__ == "__main__":
    main()
