# Прямые Python-команды воспроизведения

Все команды запускаются из корня репозитория:

```bash
cd /path/to/pinn-small-data-jcp
```

## Проверка кода

```bash
python3 -B -m unittest \
  tests.test_heat_equation \
  tests.test_two_phase_fv \
  tests.test_corrected_unified_protocol \
  tests.test_classical_tuning \
  tests.test_generator_inputs \
  tests.test_publication_defaults \
  tests.test_device_execution -v
```

## Генерация Tests 2--4

```bash
python3 -B paper_jcp/generate_corrected_benchmarks.py \
  --test all \
  --output-root paper_jcp/generated/publication_data \
  --output-steps 100 \
  --final-time-test2 0.3 \
  --final-time-test3 0.05 \
  --final-time-test4 0.2 \
  --cfl 0.2 \
  --workers 1 \
  --permeability-test2 reproducibility/data_inputs/test2/permeability.npy \
  --permeability-test3 reproducibility/data_inputs/test3/permeability.npy \
  --permeability-test4 reproducibility/data_inputs/test4/permeability.npy \
  --test4-parameter-dir reproducibility/data_inputs/test4
```

## Текущий расширенный эксперимент

По умолчанию launcher использует 10 seed (`101`--`110`) и 50,000 эпох PINN.

Test 1:

```bash
python3 -B paper_jcp/run_publication_suite.py --test test1 --part all --device cpu --workers 1
```

Test 2:

```bash
python3 -B paper_jcp/run_publication_suite.py --test test2 --part all --device cpu --workers 1
```

Test 3:

```bash
python3 -B paper_jcp/run_publication_suite.py --test test3 --part all --device cpu --workers 1
```

Test 4:

```bash
python3 -B paper_jcp/run_publication_suite.py --test test4 --part all --device cpu --workers 1
```

Все четыре теста последовательно:

```bash
python3 -B paper_jcp/run_publication_suite.py --test all --part all --device cpu --workers 1
```

PINN и classical stages можно запускать отдельно:

```bash
python3 -B paper_jcp/run_publication_suite.py --test test4 --part pinn --device cuda:0 --workers 1
python3 -B paper_jcp/run_publication_suite.py --test test4 --part classical --device cpu --workers 1
```

Проверить разворачиваемые команды без запуска обучения:

```bash
python3 -B paper_jcp/run_publication_suite.py --test all --part all --device cpu --workers 1 --dry-run
```

## Таблицы и визуализации

```bash
python3 -B paper_jcp/build_final_metrics_report.py

python3 -B paper_jcp/plot_model_comparison.py \
  --test test4 --budget 50 --seed 101 --pinn h \
  --test4-sample 0 --time-fractions 0.7,1 --device cpu
```

## Дополнительный эксперимент без labels / с 20 labels

```bash
python3 -B paper_jcp/run_zero_label_experiment.py \
  --experiment-name zero_label_test12_e50k_nr128 \
  --test all --budgets 0,20 --seeds 101,102,103,104,105,106,107,108,109,110 \
  --balancers h,ho --epochs 50000 --collocation-points 128 \
  --hidden-dim 128 --depth 6 --activation gelu \
  --learning-rate 0.003 --min-learning-rate 0.00001 \
  --l0-data 0.03 --l0-ibc 0.003 --orthogonal-floor 0.25 \
  --gradient-clip 1 --physics-validation-size 512 --log-every 100 \
  --corrected-data-root paper_jcp/generated/publication_data \
  --device cpu --workers 1
```

Для одной NVIDIA GPU заменить `--device cpu` на `--device cuda:0`; при этом
оставить `--workers 1`. Классические модели и генератор конечных объёмов в
зафиксированном протоколе исполняются на CPU.

## Графики компонент loss и их весов

```bash
python3 -B paper_jcp/plot_training_history_comparison.py
```

Скрипт автоматически читает истории `N=0,20` из
`zero_label_test12_e50k_nr128` и истории `N=20,50,100,200,500,1000` из
финальных экспериментов Tests 1--2. Альтернативные каталоги можно передать
через параметры, перечисленные в `--help`.

## Компиляция статьи

```bash
cd paper_jcp/jcp_article
python3 build_article_assets.py
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Проверка воспроизведённых метрик

После завершения всех стадий:

```bash
python3 -B reproducibility/verify_results.py --suite all
```

По умолчанию используются `rtol=1e-5` и `atol=1e-7`. Для проверки на том же
CPU и программном окружении допуски можно ужесточить через `--rtol` и
`--atol`.
