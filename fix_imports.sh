#!/usr/bin/env bash
#
# fix_imports.sh — доводчик импортов после reorganize.sh.
# Причина бага: BSD sed на macOS не понимает \b (границу слова),
# поэтому замены в reorganize.sh молча не применились.
# Здесь явные паттерны без \b — работает и на macOS, и на Linux.
#
# Идемпотентен: повторный запуск ничего не ломает (старых паттернов
# уже не останется). Запускать из КОРНЯ проекта.
#
set -euo pipefail

if [ ! -d .git ]; then
  echo "!!! Нет .git. Останови, закоммить текущее состояние, потом запусти."
  exit 1
fi

# Детект GNU vs BSD sed -> правильный флаг -i
if sed --version >/dev/null 2>&1; then
  SED_INPLACE=(-i)        # GNU
else
  SED_INPLACE=(-i '')     # BSD/macOS
fi

# Порядок ВАЖЕН: длинные имена раньше коротких, иначе
# 'engine' заденет 'engine_portfolio', 'strategies' заденет 'strategies_*'.
# Паттерны привязаны к 'from <m> import' и 'import <m>' с пробелом/концом —
# это не трогает уже переписанные 'from core.engine' (там после точки нет пробела перед именем).
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
  "strategies_donchian_champions|strategies.strategies_donchian_champions"
  "strategies|strategies.strategies"
  "not_now|strategies.strategies_donchian_champions"
)

echo ">>> Переписываю импорты во всех .py (кроме venv, .git, __pycache__)"
while IFS= read -r -d '' file; do
  for rule in "${RULES[@]}"; do
    old="${rule%%|*}"; new="${rule##*|}"
    # 'from <old> import'  ->  'from <new> import'
    sed "${SED_INPLACE[@]}" -E "s/^([[:space:]]*from )${old}( import)/\1${new}\2/g" "$file"
    # 'import <old>' в начале строки (с отступом)  ->  'import <new>'
    sed "${SED_INPLACE[@]}" -E "s/^([[:space:]]*import )${old}([[:space:]]|$)/\1${new}\2/g" "$file"
  done
done < <(find core strategies runners diagnostics -name '*.py' -print0)

echo ">>> Проверка остатков (должно быть пусто):"
LEFT=$(grep -rn "from engine import\|from strategies import\|from not_now import\|from display import\|from universe import" \
       core strategies runners diagnostics 2>/dev/null || true)
if [ -z "$LEFT" ]; then
  echo "    чисто — старых импортов не осталось"
else
  echo "$LEFT"
  echo "    ^ остались, разберём вручную"
fi

git add -A
echo ">>> Готово. Теперь проверь запуск (из корня):"
echo "      python -m runners.dev_test"
echo "      python -m runners.run_dualmom"
echo "      python -m runners.compare_all"