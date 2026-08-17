"""
Ejemplos de uso de risk_metrics con datos reales de Yahoo Finance.
"""

from risk_metrics import (
    drawdown,
    fetch_prices,
    returns_from_prices,
    summary_stats,
)


def ejemplo_metricas_basicas():
    """Descarga precios reales y calcula el resumen completo de métricas."""
    print("=" * 70)
    print("EJEMPLO 1: MÉTRICAS DE RIESGO/RETORNO (datos reales)")
    print("=" * 70)

    prices = fetch_prices(["AAPL", "MSFT", "SPY"], start="2021-01-01")
    returns = returns_from_prices(prices)

    stats = summary_stats(returns, riskfree_rate=0.03)
    print("\n" + stats.round(4).to_string())


def ejemplo_drawdown():
    """Analiza la evolución del drawdown de un solo ticker."""
    print("\n" + "=" * 70)
    print("EJEMPLO 2: ANÁLISIS DE DRAWDOWN")
    print("=" * 70)

    prices = fetch_prices(["SPY"], start="2020-01-01")
    returns = returns_from_prices(prices["SPY"])

    dd = drawdown(returns)
    peor_dia = dd["Drawdown"].idxmin()

    print(f"\nMáximo drawdown: {dd['Drawdown'].min():.2%}")
    print(f"Fecha del mínimo: {peor_dia.date()}")

    # Descomenta para graficar (requiere matplotlib):
    # dd["Drawdown"].plot(title="Drawdown SPY")
    # import matplotlib.pyplot as plt
    # plt.show()


def ejemplo_comparar_carteras():
    """Compara métricas de riesgo entre distintos ETFs sectoriales."""
    print("\n" + "=" * 70)
    print("EJEMPLO 3: COMPARACIÓN DE ETFs SECTORIALES")
    print("=" * 70)

    tickers = ["XLK", "XLF", "XLE", "XLV"]  # Tech, Financials, Energy, Health
    prices = fetch_prices(tickers, start="2021-01-01")
    returns = returns_from_prices(prices)

    stats = summary_stats(returns, riskfree_rate=0.03)
    ranking = stats.sort_values("Sharpe Ratio", ascending=False)

    print("\nRanking por Sharpe Ratio:")
    print(ranking[["Annualized Return", "Annualized Vol", "Sharpe Ratio"]].round(4).to_string())


def main():
    ejemplo_metricas_basicas()
    ejemplo_drawdown()
    ejemplo_comparar_carteras()


if __name__ == "__main__":
    main()
