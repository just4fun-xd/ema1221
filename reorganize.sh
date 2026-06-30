#!/usr/bin/env bash
#
# reorganize.sh — навести порядок в проекте EMA1221.
# Папочная структура + __init__.py + переписанные импорты.
#
# БЕЗОПАСНОСТЬ:
#   - использует git mv (история файлов сохраняется)
#   - НЕ запускай без git: сначала `git init && git add -A && git commit -m wip`
#   - идемпотентен: повторный запуск не ломает уже перемещённое
#   - после прогона ОБЯЗАТЕЛЬНО проверь импорты (чек-лист в конце вывода)
#
set -euo pipefail

echo ">>> 0. Проверка, что мы в git-репозитории"
if [ ! -d .git ]; then
  echo "!!! Нет .git. Сначала: git init && git add -A && git commit -m 'before reorg'"
  exit 1
fi

# Хелпер: git mv только если источник существует и ещё не на месте
move() {
  local src="$1" dst="$2"
  if [ -e "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    git mv -k "$src" "$dst" 2>/dev/null && echo "  moved $src -> $dst" || true
  fi
}

echo ">>> 1. Создаю папки"
mkdir -p core strategies runners diagnostics results scripts docs archive

echo ">>> 2. Удаляю первые пробы пера"
for f in step1.py test.py backtest.py; do
  if [ -e "$f" ]; then git rm -q "$f" && echo "  removed $f"; fi
done

echo ">>> 3. Переименовываю not_now.py -> champions"
move "not_now.py" "strategies/strategies_donchian_champions.py"

echo ">>> 4. Ядро и движки -> core/"
move "engine.py"            "core/engine.py"
move "engine_portfolio.py"  "core/engine_portfolio.py"
move "display.py"           "core/display.py"
move "universe.py"          "core/universe.py"

echo ">>> 5. Стратегии -> strategies/"
for f in strategies strategies_turtle strategies_seasonal strategies_bollinger \
         strategies_meanrev strategies_pairs strategies_pairs_kalman strategies_dualmom; do
  move "${f}.py" "strategies/${f}.py"
done

echo ">>> 6. Runner-ы -> runners/"
for f in main dev_test compare_all run_dualmom run_wide tilt_yearly \
         run_pairs run_pairs_kalman run_pairs_yearly main_ema main_donchian; do
  move "${f}.py" "runners/${f}.py"
done

echo ">>> 7. Диагностика -> diagnostics/"
move "_concentration_check.py" "diagnostics/_concentration_check.py"
move "pair_diagnostic.py"      "diagnostics/pair_diagnostic.py"

echo ">>> 8. Результаты -> results/"
for f in result.txt result_donchian.txt result_ema.txt result_years_ema.txt; do
  move "$f" "results/$f"
done

echo ">>> 9. Скрипты данных -> scripts/"
move "download_databento.py" "scripts/download_databento.py"
move "check_databento.py"    "scripts/check_databento.py"

echo ">>> 10. Документация -> docs/"
for f in *.md; do
  [ -e "$f" ] || continue
  if [ "$f" != "README.md" ]; then move "$f" "docs/$f"; fi
done

echo ">>> 11. __init__.py в пакеты"
for pkg in core strategies runners diagnostics; do
  touch "$pkg/__init__.py"
  git add "$pkg/__init__.py"
  echo "  $pkg/__init__.py"
done

echo ">>> 12. Переписываю импорты во ВСЕХ .py (sed)"
# Каждый старый модуль -> новый пакетный путь.
# not_now переименован, поэтому отдельное правило.
declare -a RULES=(
  "engine_portfolio|core.engine_portfolio"
  "engine|core.engine"
  "display|core.display"
  "universe|core.universe"
  "strategies_turtle|strategies.strategies_turtle"
  "strategies_seasonal|strategies.strategies_seasonal"
  "strategies_bollinger|strategies.strategies_bollinger"
  "strategies_meanrev|strategies.strategies_meanrev"
  "strategies_pairs_kalman|strategies.strategies_pairs_kalman"
  "strategies_pairs|strategies.strategies_pairs"
  "strategies_dualmom|strategies.strategies_dualmom"
  "strategies|strategies.strategies"
  "not_now|strategies.strategies_donchian_champions"
)
# ВАЖНО: порядок правил — длинные имена раньше коротких,
# иначе 'engine' заденет 'engine_portfolio'. Список уже упорядочен.
while IFS= read -r -d '' file; do
  for rule in "${RULES[@]}"; do
    old="${rule%%|*}"; new="${rule##*|}"
    # 'from <old> import'  и  'import <old>'
    sed -i.bak -E "s/\bfrom ${old} import/from ${new} import/g; s/\bimport ${old}\b/import ${new}/g" "$file"
  done
  rm -f "${file}.bak"
done < <(find core strategies runners diagnostics -name '*.py' -print0)

echo ">>> 13. Стейджу изменения"
git add -A

cat << 'NOTE'

============================================================
ГОТОВО. Но импорты надо проверить — sed покрывает известные
модули, но не экзотические случаи (relative import, importlib,
строковые имена). ЧЕК-ЛИСТ:

  1. Запусти каждый runner и убедись, что импорты живые:
       python -m runners.run_dualmom
       python -m runners.compare_all
       python -m runners.dev_test
     (запуск через -m, потому что теперь это пакеты!)

  2. Если ловишь ModuleNotFoundError — найди остаток:
       grep -rn "import engine\b\|from engine\b" .
       grep -rn "not_now" .

  3. Когда всё зелёное:
       git commit -m "reorganize project into packages"

  4. Если что-то сломалось и хочешь откатить:
       git reset --hard HEAD   (вернёт к коммиту 'before reorg')
============================================================
NOTE