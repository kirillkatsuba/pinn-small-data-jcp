# Сравнительные визуализации GT / PINN / XGB / SVM / GP

Скрипт `plot_model_comparison.py` строит сравнения на тех же размеченных
точках, бюджетах и seed, которые использовались в итоговом эксперименте.

- PINN загружается из final checkpoint и не переобучается.
- XGB, SVM и GP один раз восстанавливаются из того же nested label set с тем
  же подбором гиперпараметров. Затем fitted-модели читаются из cache.
- Test 1 использует standard/random tuning, соответствующий его CSV.
- Tests 2--4 используют strong/time-aware CV.
- По умолчанию используются H-PINN, `N=50`, seed `101`.

## Test 1: профили решения

```bash
cd /path/to/pinn-small-data-jcp

python3 -B paper_jcp/plot_model_comparison.py \
  --test test1 \
  --budget 50 \
  --seed 101 \
  --pinn h \
  --time-fractions 0,0.1,0.25,0.5,0.7,1
```

## Test 2: pressure и насыщенности

```bash
python3 -B paper_jcp/plot_model_comparison.py \
  --test test2 \
  --budget 50 \
  --seed 101 \
  --pinn h \
  --time-fractions 0,0.5,0.7,1
```

## Test 3

```bash
python3 -B paper_jcp/plot_model_comparison.py \
  --test test3 \
  --budget 50 \
  --seed 101 \
  --pinn h \
  --time-fractions 0,0.5,0.7,1
```

## Test 4: параметрическая задача

```bash
python3 -B paper_jcp/plot_model_comparison.py \
  --test test4 \
  --budget 50 \
  --seed 101 \
  --pinn h \
  --test4-sample 0 \
  --time-fractions 0,0.5,0.7,1
```

`--test4-sample` выбирает один из 100 наборов физических параметров. Для
проверки устойчивости полезно дополнительно построить samples `25`, `50`,
`75` и `99`.

## Структура рисунков

Для каждого времени Tests 2--4 создаётся один рисунок:

- строки: Pressure, `S_w`, `S_o`;
- столбцы: GT, PINN, XGB, SVM, GP;
- внутри каждой строки используется общая шкала цветов;
- saturation показывается на физической шкале `[0, 1]`.

Рисунки сохраняются в
`paper_jcp/figures/final_model_comparison/<test>/`, а fitted classical models —
в `paper_jcp/experiments/qualitative_classical_cache/`.

Для принудительного повторного подбора добавьте `--force-refit-classical`.
Для HO-PINN замените `--pinn h` на `--pinn ho`.

Для основной статьи рекомендуется показать `N=50` и два временных среза:
`0.7` (граница интерполяции) и `1.0` (финальная экстраполяция). Test 4 следует
показать минимум на двух параметрических samples.
