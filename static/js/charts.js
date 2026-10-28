/**
 * WeatherWise Interactive Visualization Engine (Plotly.js)
 * Clean, lightweight, professional modern charts adapted to the light/atmospheric theme.
 */

window.WeatherWiseCharts = (function() {
  'use strict';

  function getBaseLayout() {
    return {
      paper_bgcolor: 'transparent',
      plot_bgcolor: 'transparent',
      font: {
        family: 'Plus Jakarta Sans, system-ui, sans-serif',
        color: '#475569',
        size: 11
      },
      margin: { t: 25, r: 20, b: 40, l: 45 },
      xaxis: {
        gridcolor: 'rgba(15, 23, 42, 0.06)',
        linecolor: 'rgba(15, 23, 42, 0.1)',
        tickcolor: 'rgba(15, 23, 42, 0.1)',
        tickfont: { color: '#64748b', size: 10 }
      },
      yaxis: {
        gridcolor: 'rgba(15, 23, 42, 0.06)',
        linecolor: 'rgba(15, 23, 42, 0.1)',
        tickcolor: 'rgba(15, 23, 42, 0.1)',
        tickfont: { color: '#64748b', size: 10 }
      },
      legend: {
        bgcolor: 'rgba(255, 255, 255, 0.75)',
        bordercolor: 'rgba(15, 23, 42, 0.08)',
        borderwidth: 1,
        font: { color: '#1e293b', size: 11 }
      },
      hoverlabel: {
        bgcolor: '#0f172a',
        bordercolor: '#0284c7',
        font: { color: '#ffffff', size: 12 }
      }
    };
  }

  /**
   * Render Time-Series Trend Chart with Moving Average and Linear Trendline
   */
  function renderTrendChart(divId, data) {
    const el = document.getElementById(divId);
    if (!el || !window.Plotly) return;

    const base = getBaseLayout();

    // Actual observations
    const traceActual = {
      x: data.dates,
      y: data.values,
      name: `Observed ${data.variable.charAt(0).toUpperCase() + data.variable.slice(1)}`,
      type: 'scatter',
      mode: 'lines',
      line: {
        color: '#0284c7',
        width: 2.2
      },
      fill: 'tozeroy',
      fillcolor: 'rgba(2, 132, 199, 0.08)'
    };

    // Moving average
    const traceMA = {
      x: data.dates,
      y: data.moving_average,
      name: 'Moving Average',
      type: 'scatter',
      mode: 'lines',
      line: {
        color: '#f59e0b',
        width: 2,
        dash: 'dash'
      }
    };

    // Regression trendline
    const traceTrend = {
      x: data.dates,
      y: data.trend_line,
      name: `Linear Trend (${data.trend_direction})`,
      type: 'scatter',
      mode: 'lines',
      line: {
        color: '#64748b',
        width: 1.8,
        dash: 'dot'
      }
    };

    const layout = {
      ...base,
      yaxis: {
        ...base.yaxis,
        title: { text: data.unit ? `${data.variable} (${data.unit})` : data.variable }
      }
    };

    Plotly.react(el, [traceActual, traceMA, traceTrend], layout, { responsive: true, displayModeBar: false });
  }

  /**
   * Render 12-Month Climatological Bar Chart
   */
  function renderMonthlyChart(divId, data) {
    const el = document.getElementById(divId);
    if (!el || !window.Plotly || !data.monthly_summary) return;

    const base = getBaseLayout();
    const months = data.monthly_summary.map(m => m.month);
    const temps = data.monthly_summary.map(m => m.avg_temp);
    const rain = data.monthly_summary.map(m => m.avg_rainfall);

    const traceTemp = {
      x: months,
      y: temps,
      name: 'Mean Temp (°C)',
      type: 'bar',
      marker: {
        color: '#0284c7',
        opacity: 0.85
      }
    };

    const traceRain = {
      x: months,
      y: rain,
      name: 'Avg Rainfall (mm)',
      type: 'bar',
      marker: {
        color: '#38bdf8',
        opacity: 0.65
      }
    };

    const layout = {
      ...base,
      barmode: 'group',
      yaxis: {
        ...base.yaxis,
        title: { text: 'Monthly Values (°C / mm)' }
      }
    };

    Plotly.react(el, [traceTemp, traceRain], layout, { responsive: true, displayModeBar: false });
  }

  /**
   * Render 12-Month Climate Projection Chart with 95% Confidence Bounds
   */
  function renderClimateChart(divId, data) {
    const el = document.getElementById(divId);
    if (!el || !window.Plotly) return;

    const base = getBaseLayout();

    const traceUpper = {
      x: data.future_months,
      y: data.upper_bounds,
      name: '95% Upper Bound',
      type: 'scatter',
      mode: 'lines',
      marker: { color: '#94a3b8' },
      line: { width: 0 },
      showlegend: false
    };

    const traceLower = {
      x: data.future_months,
      y: data.lower_bounds,
      name: 'Confidence Interval (±2.5°C)',
      type: 'scatter',
      mode: 'lines',
      fill: 'tonexty',
      fillcolor: 'rgba(2, 132, 199, 0.12)',
      line: { width: 0 }
    };

    const traceProjection = {
      x: data.future_months,
      y: data.projected_temps,
      name: 'Projected Mean Temp (°C)',
      type: 'scatter',
      mode: 'lines+markers',
      line: {
        color: '#0284c7',
        width: 2.8
      },
      marker: {
        size: 6,
        color: '#0284c7'
      }
    };

    const layout = {
      ...base,
      yaxis: {
        ...base.yaxis,
        title: { text: 'Projected Temperature (°C)' }
      }
    };

    Plotly.react(el, [traceUpper, traceLower, traceProjection], layout, { responsive: true, displayModeBar: false });
  }

  return {
    renderTrendChart,
    renderMonthlyChart,
    renderClimateChart
  };
})();
