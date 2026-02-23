<div class="hero" markdown>

# pyscf_auto

PySCF/ASE 기반 양자화학 계산 워크플로우 도구.
ORCA 스타일 `.inp` 파일로 실험을 정의하고, 자동 재시도와 텔레그램 알림을 받으세요.

</div>

<div class="lang-buttons" markdown>

[**한국어** 매뉴얼](ko/index.md){ .btn-primary }
[**English** Manual](en/index.md){ .btn-outline }

</div>

---

<div class="grid" markdown>

<div class="card" markdown>

### :material-file-edit: .inp 파일 입력

ORCA 스타일 문법으로 계산 조건을 정의합니다.
`! Opt B3LYP def2-SVP D3BJ PCM(water)`

</div>

<div class="card" markdown>

### :material-refresh: 자동 재시도

SCF 수렴 실패 시 5단계 전략으로 자동 재시도합니다.
Level shift, DIIS 변경, damping 등.

</div>

<div class="card" markdown>

### :material-send: 텔레그램 알림

계산 시작, 완료, 실패, 하트비트를
텔레그램으로 실시간 알림받으세요.

</div>

<div class="card" markdown>

### :material-folder-outline: 자동 정리

완료된 계산을 분자/작업 유형별로
자동 정리합니다 (Hill notation).

</div>

</div>

---

## Quick Start

```bash
# 설치
conda create -n pyscf_auto -c daehyupsohn -c conda-forge pyscf_auto
conda activate pyscf_auto

# .inp 파일 작성 후 실행
pyscf_auto run-inp --reaction-dir ~/pyscf_runs/my_calc

# 상태 확인
pyscf_auto status --reaction-dir ~/pyscf_runs/my_calc
```
