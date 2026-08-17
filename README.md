# Portfolio Risk Metrics

Librería en Python con métricas estándar de riesgo y retorno para series de
activos financieros — retorno anualizado, volatilidad, Sharpe ratio,
drawdown, skewness/kurtosis, y Value at Risk (histórico y ajustado por
Cornish-Fisher). Los datos se descargan en vivo desde Yahoo Finance.

## Instalación

```bash
git clone https://github.com/Gustavo-Campos-Luna/portfolio-risk-metrics.git
cd portfolio-risk-metrics
pip install -r requirements.txt
```

## Uso

```python
from risk_metrics import fetch_prices, returns_from_prices, summary_stats

prices = fetch_prices(["AAPL", "MSFT", "SPY"], start="2021-01-01")
returns = returns_from_prices(prices)

summary_stats(returns, riskfree_rate=0.03)
```

```
        Annualized Return  Annualized Vol  Sharpe Ratio  ...  Max Drawdown
Ticker
AAPL               0.1672          0.2786        0.4780  ...       -0.3336
MSFT               0.1838          0.2571        0.5807  ...       -0.3715
SPY                0.1472          0.1711        0.6653  ...       -0.2450
```

Más ejemplos (análisis de drawdown, comparación de ETFs sectoriales) en
[`examples/basic_usage.py`](examples/basic_usage.py):

```bash
PYTHONPATH=. python examples/basic_usage.py
```

## Métricas incluidas

| Función | Qué calcula |
|---|---|
| `annualized_return` | CAGR implícito de la serie de retornos |
| `annualized_vol` | Desviación estándar anualizada |
| `sharpe_ratio` | Retorno en exceso sobre la tasa libre de riesgo, por unidad de volatilidad |
| `drawdown` / `max_drawdown` | Caída porcentual desde el máximo histórico (peak-to-trough) |
| `skewness` / `kurtosis` | Momentos 3° y 4° de la distribución de retornos |
| `semideviation` | Desviación estándar solo de los retornos bajo la media (riesgo a la baja) |
| `var_historic` / `cvar_historic` | VaR y Expected Shortfall al percentil elegido |
| `var_cornish_fisher` | VaR gaussiano ajustado por skewness/kurtosis (Cornish & Fisher, 1938) |

Fórmulas y referencias completas en los docstrings de
[`risk_metrics/metrics.py`](risk_metrics/metrics.py).

## Estructura

```
risk_metrics/
  data.py       fetch_prices() — descarga vía yfinance
  metrics.py    funciones de riesgo/retorno (ver tabla arriba)
tests/
  test_metrics.py   21 tests: valores conocidos y casos borde
examples/
  basic_usage.py
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

## Limitaciones

- Los retornos se tratan como observaciones independientes; no hay ajuste
  por autocorrelación serial.
- `var_cornish_fisher` puede producir resultados poco intuitivos con
  skewness/kurtosis extremos — es una aproximación, no un modelo exacto de
  la cola de la distribución.
- Sin manejo de dividendos más allá del ajuste que ya aplica `yfinance`
  (`auto_adjust=True`).
- Depende de la disponibilidad de datos en Yahoo Finance.

## Notas de desarrollo

Los conceptos (retornos ajustados al riesgo, VaR histórico y
Cornish-Fisher, drawdown) se aprendieron en el curso "Introduction to
Portfolio Construction and Analysis with Python" (Coursera/EDHEC). El
código de este repositorio es una reconstrucción propia con asistencia de
IA: usa datos reales de Yahoo Finance en vez de los CSV estáticos del
curso, está organizado como paquete con tests (21 casos, incluyendo bordes
como volatilidad cero y series vacías), y no reutiliza el código entregado
en el curso. Las fórmulas mismas son estándar de la literatura de gestión
de riesgo, no exclusivas de ningún curso.

## Licencia

[MIT](LICENSE)
