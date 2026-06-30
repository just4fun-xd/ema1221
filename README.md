# EMA1221 — Systematic Trading Research

Исследовательский проект по систематическим торговым стратегиям.
Минималистичный стек (pandas/numpy/matplotlib + yfinance), без backtrader
и vectorbt — каждая строка логики понятна механически.

Цель: ~одна стратегия в неделю, с приоритетом на статистическую строгость,
воспроизводимый back/forward-тест и честную документацию ограничений.
Жёсткое требование: максимальная просадка < 40%.

## Структура

```
EMA1221/
  core/           движки и инфраструктура
    engine.py             посерийный движок (один инструмент -> позиция)
    engine_portfolio.py   портфельный движок (матрица весов, cross-sectional)
    display.py            форматирование таблиц вывода
    universe.py           загрузка широкой корзины (S&P 500)
  strategies/     библиотека стратегий (close -> position, или prices -> weights)
    strategies.py                     EMA-семейство (cross, ensemble, vol-target)
    strategies_turtle.py              Donchian breakout / pyramid
    strategies_donchian_champions.py  лучшие Donchian-варианты (бывш. not_now)
    strategies_dualmom.py             cross-sectional dual momentum + варианты шорта
    strategies_seasonal.py            сезонные (natural gas и др.)
    strategies_bollinger.py           Bollinger %b / squeeze
    strategies_meanrev.py             mean-reversion
    strategies_pairs.py               pairs trading (z-score)
    strategies_pairs_kalman.py        pairs trading (Kalman)
  runners/        исполняемые сценарии бэктестов
    main.py, dev_test.py              основные прогоны по корзинам
    compare_all.py                    cross-sec vs per-series на одной корзине
    run_dualmom.py                    dual momentum + три теста честности
    run_wide.py                       тест концентрации на S&P 500
    tilt_yearly.py                    поэлементный годовой отчёт для tilt
    run_pairs*.py                     прогоны pairs
  diagnostics/    диагностические скрипты
    _concentration_check.py           концентрация P&L по именам
    pair_diagnostic.py                диагностика пар
  results/        сохранённые выводы прогонов (*.txt)
  scripts/        вспомогательные скрипты (загрузка данных Databento)
  data/           данные (не в git)
  docs/           отчёты, cheat sheets, документация гипотез
```

## Запуск

После реорганизации в пакеты runner-ы запускаются как модули из корня:

```bash
python -m runners.run_dualmom
python -m runners.compare_all
python -m runners.dev_test
```

Установка зависимостей: `pip install -r requirements.txt`

## Статусы стратегий

| Трек | Статус | Вывод |
|---|---|---|
| EMA Ensemble + Vol Target | ✅ основной кандидат | DD < 40% (worst −23.3%); работает на акциях, не на commodities |
| Donchian 4-step pyramid | ✅ лидер на commodities | `Don Est+macd+4step+take` — лучший на сырье |
| Bollinger (%b, squeeze) | ⚪ закрыт | %b без независимого сигнала; squeeze переобучен |
| Seasonal | ⚪ документирован | natural gas и варианты |
| Pairs (z-score / Kalman) | ⚪ закрыт | нет edge даже при нулевых издержках; структурные сломы |
| Carry / term structure | ⚪ закрыт | нужно 20-30 инструментов; провал на 7 |
| Dual Momentum (short leg) | ⚪ исследование | cross-sectional шорт — хедж, не альфа; см. docs/SHORT_RESULTS.md |

Подробности и закрытые гипотезы — в `docs/` (PROJECT_SUMMARY, BENCHMARK_RESULTS,
SHORT_RESULTS, PAIRS_RESULTS, cheat sheets).

## Принципы

- Edge должен существовать в сигнале, подходить инструменту и переживать издержки.
- Каждая стратегия: механика -> реализация -> бэктест -> проверка стабильности
  параметров -> явное решение «закрыть/развивать» -> документация.
- Медиана вместо среднего для агрегатов (выбросы вроде Tesla/Nvidia искажают).
- Закрытые гипотезы документируются с причинами — честная карта ограничений.

## Данные

- yfinance дневные бары (основное; акции чистые, без roll-проблемы).
- Databento CME continuous futures (`futures_m1_m2_fixed.csv`) для carry.
- Continuous futures требуют roll-склейки, иначе ложные сигналы на роллах.