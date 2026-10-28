/**
 * WeatherWise Application Logic
 * Coordinates live Open-Meteo API fetching, real historical datasets,
 * dynamic atmospheric theme adaptation, sub-tab analytics, and lifestyle preparedness.
 */

(function() {
  'use strict';

  const state = {
    city: 'London',
    lat: null,
    lon: null,
    isGPS: false,
    mode: 'live', // 'live' or 'demo'
    currentVibe: 'rainy',
    trendsPeriod: '30d',
    trendsVariable: 'temperature',
    anomalyVariable: 'temperature',
    anomalyMethod: 'iqr',
    activeTab: 'tab-today',
    activeSubtab: 'subtab-trends',
    autoRefreshTimer: null
  };

  // ==========================================================================
  // INITIALIZATION
  // ==========================================================================

  async function init() {
    initAtmosphericFX();
    setupTabListeners();
    setupSubtabListeners();
    setupTrendsControls();
    setupAnomalyControls();
    setupModeToggle();
    setupRealtimeControls();
    await loadCities();
    await autoDetectUserLocation();
    await refreshAllWeatherData();
    setupAutoRefresh();
  }

  // ==========================================================================
  // DYNAMIC ATMOSPHERIC THEME ENGINE & BACKGROUND FX
  // ==========================================================================

  let lightningInterval = null;

  function initAtmosphericFX() {
    // 1. Initialize Rain Particles
    const rainContainer = document.getElementById('ww-fx-rain');
    if (rainContainer && rainContainer.children.length === 0) {
      const dropCount = 50;
      const frag = document.createDocumentFragment();
      for (let i = 0; i < dropCount; i++) {
        const drop = document.createElement('div');
        drop.className = 'ww-raindrop';
        const leftPercent = (i * (100 / dropCount)) + (Math.random() * 2 - 1);
        const duration = (0.65 + Math.random() * 0.45).toFixed(2);
        const delay = (Math.random() * 2.2).toFixed(2);
        const opacity = (0.35 + Math.random() * 0.45).toFixed(2);
        const height = Math.floor(45 + Math.random() * 35);

        drop.style.left = `${leftPercent}%`;
        drop.style.animationDuration = `${duration}s`;
        drop.style.animationDelay = `${delay}s`;
        drop.style.opacity = opacity;
        drop.style.height = `${height}px`;
        frag.appendChild(drop);
      }
      rainContainer.appendChild(frag);
    }

    // 2. Initialize Snow Particles
    const snowContainer = document.getElementById('ww-fx-snow');
    if (snowContainer && snowContainer.children.length === 0) {
      const flakeCount = 35;
      const frag = document.createDocumentFragment();
      for (let i = 0; i < flakeCount; i++) {
        const flake = document.createElement('div');
        flake.className = 'ww-snowflake';
        const leftPercent = (i * (100 / flakeCount)) + (Math.random() * 2 - 1);
        const duration = (4.0 + Math.random() * 4.5).toFixed(2);
        const delay = (Math.random() * 4.0).toFixed(2);
        const size = Math.floor(5 + Math.random() * 6);
        const swayDuration = (2.2 + Math.random() * 2.0).toFixed(2);

        flake.style.left = `${leftPercent}%`;
        flake.style.width = `${size}px`;
        flake.style.height = `${size}px`;
        flake.style.animationDuration = `${duration}s, ${swayDuration}s`;
        flake.style.animationDelay = `${delay}s, ${delay}s`;
        flake.style.opacity = (0.4 + Math.random() * 0.5).toFixed(2);
        frag.appendChild(flake);
      }
      snowContainer.appendChild(frag);
    }
  }

  function updateAtmosphericFX(vibe) {
    const bolt = document.getElementById('ww-lightning-bolt');
    if (lightningInterval) {
      clearInterval(lightningInterval);
      lightningInterval = null;
    }

    if (vibe === 'stormy') {
      lightningInterval = setInterval(() => {
        if (bolt) {
          const randLeft = Math.floor(15 + Math.random() * 65);
          bolt.style.left = `${randLeft}%`;
        }
      }, 7500);
    }
  }

  function setWeatherVibe(vibe) {
    if (!vibe) return;
    state.currentVibe = vibe;
    document.body.setAttribute('data-vibe', vibe);
    updateAtmosphericFX(vibe);
  }

  // ==========================================================================
  // LIVE VS DEMO MODE TOGGLE
  // ==========================================================================

  function setupModeToggle() {
    const btn = document.getElementById('btn-toggle-mode');
    const lbl = document.getElementById('lbl-mode');
    if (!btn || !lbl) return;

    btn.addEventListener('click', () => {
      state.mode = (state.mode === 'live') ? 'demo' : 'live';
      if (state.mode === 'demo') {
        lbl.textContent = 'Demo Simulation Mode';
        btn.classList.add('demo-active');
      } else {
        lbl.textContent = 'Live Open-Meteo API';
        btn.classList.remove('demo-active');
      }
      refreshAllWeatherData();
    });
  }

  // ==========================================================================
  // REAL-TIME HARDWARE SENSOR CONTROLS & GEOLOCATION
  // ==========================================================================

  function setupRealtimeControls() {
    const btnSync = document.getElementById('btn-sync-now');
    const syncIcon = document.getElementById('icon-sync-spin');
    const liveStatus = document.getElementById('ww-live-status');

    if (btnSync) {
      btnSync.addEventListener('click', async () => {
        if (syncIcon) syncIcon.classList.add('spinning');
        if (liveStatus) liveStatus.textContent = 'SYNCING SENSORS...';

        try {
          await refreshAllWeatherData(true);
          if (liveStatus) liveStatus.textContent = 'LIVE SENSOR ACTIVE';
        } catch (e) {
          console.error('Manual sensor sync failed:', e);
          if (liveStatus) liveStatus.textContent = 'SYNC DELAY';
        } finally {
          setTimeout(() => {
            if (syncIcon) syncIcon.classList.remove('spinning');
            if (liveStatus) liveStatus.textContent = 'LIVE SENSOR ACTIVE';
          }, 650);
        }
      });
    }

    const btnLocation = document.getElementById('btn-detect-location');
    if (btnLocation) {
      btnLocation.addEventListener('click', () => {
        triggerGeolocation(true);
      });
    }
  }

  async function autoDetectUserLocation() {
    if (!('geolocation' in navigator)) return;
    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          await applyUserCoordinates(position.coords.latitude, position.coords.longitude);
          resolve();
        },
        () => {
          // Graceful fallback to default city if location is not granted
          resolve();
        },
        { enableHighAccuracy: true, timeout: 5000, maximumAge: 60000 }
      );
    });
  }

  function triggerGeolocation(manual = false) {
    if (!('geolocation' in navigator)) {
      if (manual) alert('Geolocation is not supported by your browser.');
      return;
    }

    const btnLocation = document.getElementById('btn-detect-location');
    if (btnLocation) {
      btnLocation.innerHTML = '<span class="spinner-border spinner-border-sm me-1" role="status"></span><span>Locating...</span>';
    }

    navigator.geolocation.getCurrentPosition(
      async (position) => {
        await applyUserCoordinates(position.coords.latitude, position.coords.longitude);
        await refreshAllWeatherData(true);
      },
      (err) => {
        console.warn('Geolocation access error:', err.message);
        if (btnLocation) {
          btnLocation.innerHTML = '<i class="bi bi-crosshair2 text-primary"></i> <span>My Location</span>';
        }
        if (manual) {
          alert('Could not retrieve your live location. Please ensure location access is allowed in your browser.');
        }
      },
      { enableHighAccuracy: true, timeout: 10000, maximumAge: 0 }
    );
  }

  async function applyUserCoordinates(lat, lon) {
    state.lat = lat;
    state.lon = lon;
    state.isGPS = true;

    try {
      const res = await fetch(`/api/reverse-geocode?lat=${lat}&lon=${lon}`);
      const data = await res.json();
      const detectedCity = data.city || 'My Location';
      state.city = detectedCity;

      const select = document.getElementById('city-select');
      if (select) {
        let gpsOption = select.querySelector('option[data-gps="true"]');
        if (!gpsOption) {
          gpsOption = document.createElement('option');
          gpsOption.setAttribute('data-gps', 'true');
          select.insertBefore(gpsOption, select.firstChild);
        }
        gpsOption.value = detectedCity;
        gpsOption.textContent = `📍 ${detectedCity} (My Location)`;
        select.value = detectedCity;
      }

      const btnLocation = document.getElementById('btn-detect-location');
      if (btnLocation) {
        btnLocation.classList.add('active');
        btnLocation.innerHTML = `<i class="bi bi-geo-alt-fill text-success"></i> <span>${detectedCity}</span>`;
      }
    } catch (err) {
      console.error('Reverse geocode error:', err);
    }
  }

  function setupAutoRefresh() {
    if (state.autoRefreshTimer) {
      clearInterval(state.autoRefreshTimer);
    }
    // Poll real-time meteorological sensors every 60 seconds
    state.autoRefreshTimer = setInterval(async () => {
      console.log('[WeatherWise Live] Auto-syncing real-time sensor stream...');
      await Promise.all([
        fetchCurrentWeather(true),
        fetchWeeklyForecast(true)
      ]);
    }, 60000);
  }

  // ==========================================================================
  // CITY SELECTION
  // ==========================================================================

  async function loadCities() {
    try {
      const res = await fetch('/api/cities');
      const data = await res.json();
      const select = document.getElementById('city-select');
      if (!select || !data.cities) return;

      select.innerHTML = '';
      data.cities.forEach(c => {
        const opt = document.createElement('option');
        opt.value = c.name;
        opt.textContent = `${c.name}, ${c.country}`;
        if (c.name === state.city) opt.selected = true;
        select.appendChild(opt);
      });

      select.addEventListener('change', (e) => {
        const val = e.target.value;
        const selectedOpt = select.options[select.selectedIndex];
        if (selectedOpt && selectedOpt.getAttribute('data-gps') === 'true') {
          state.isGPS = true;
        } else {
          state.isGPS = false;
          state.lat = null;
          state.lon = null;
          const btnLocation = document.getElementById('btn-detect-location');
          if (btnLocation) {
            btnLocation.classList.remove('active');
            btnLocation.innerHTML = '<i class="bi bi-crosshair2 text-primary"></i> <span>My Location</span>';
          }
        }
        state.city = val;
        refreshAllWeatherData(true);
      });
    } catch (err) {
      console.error('Failed to load cities:', err);
    }
  }

  // ==========================================================================
  // DATA REFRESH ORCHESTRATION
  // ==========================================================================

  async function refreshAllWeatherData(forceRefresh = false) {
    await Promise.all([
      fetchCurrentWeather(forceRefresh),
      fetchWeeklyForecast(forceRefresh),
      fetchTrendsData(),
      fetchMonthlyData(),
      fetchSeasonsData(),
      fetchAnomaliesData(),
      fetchAccuracyData(),
      fetchClimateData(),
      fetchPreparednessData()
    ]);
  }

  // ==========================================================================
  // TAB 1: CURRENT LIVE WEATHER & 24H HOURLY
  // ==========================================================================

  async function fetchCurrentWeather(forceRefresh = false) {
    try {
      let url = `/api/weather/current?city=${encodeURIComponent(state.city)}&mode=${state.mode}`;
      if (state.isGPS && state.lat !== null && state.lon !== null) {
        url += `&lat=${state.lat}&lon=${state.lon}`;
      }
      if (forceRefresh) {
        url += '&refresh=true';
      }
      const res = await fetch(url);
      const data = await res.json();
      if (!data || data.error) return;

      // Automatically transition theme to current weather
      setWeatherVibe(data.condition_vibe);

      // Data Source Indicator
      const srcBadge = document.getElementById('badge-data-source');
      if (srcBadge) {
        if (data.is_live) {
          srcBadge.className = 'badge bg-success bg-opacity-10 text-success border border-success border-opacity-25 px-2 py-1';
          srcBadge.innerHTML = '<i class="bi bi-broadcast"></i> Live Open-Meteo Station';
        } else {
          srcBadge.className = 'badge bg-warning bg-opacity-10 text-warning border border-warning border-opacity-25 px-2 py-1';
          srcBadge.innerHTML = '<i class="bi bi-cpu"></i> Demo / Simulated Data';
        }
      }

      const syncEl = document.getElementById('lbl-last-sync');
      if (syncEl) syncEl.textContent = `Synced at ${data.last_updated}`;

      // Hero Card values
      const locationBadge = document.getElementById('weather-location-badge');
      if (locationBadge) locationBadge.textContent = `${data.city}, ${data.country} · ${data.climate}`;

      const tempVal = document.getElementById('current-temp-val');
      if (tempVal) tempVal.textContent = Math.round(data.temperature);

      const conditionText = document.getElementById('current-condition-text');
      if (conditionText) conditionText.textContent = data.condition_name;

      const conditionDesc = document.getElementById('current-condition-desc');
      if (conditionDesc) conditionDesc.textContent = data.description;

      const highLow = document.getElementById('vital-high-low');
      if (highLow) highLow.textContent = `${data.high_temp}° / ${data.low_temp}°`;

      const pressure = document.getElementById('vital-pressure');
      if (pressure) pressure.textContent = `${data.pressure} hPa`;

      // Vitals
      const feelsLike = document.getElementById('vital-feels-like');
      if (feelsLike) feelsLike.textContent = `${data.feels_like}°C`;

      const humidity = document.getElementById('vital-humidity');
      if (humidity) humidity.textContent = `${data.humidity}%`;

      const wind = document.getElementById('vital-wind');
      if (wind) wind.textContent = `${data.wind_speed} km/h`;

      const uv = document.getElementById('vital-uv');
      if (uv) uv.textContent = `${data.uv_index} of 11`;

      // 24h Hourly Scroller
      renderHourlyScroller(data.hourly);
    } catch (err) {
      console.error('Error fetching current weather:', err);
    }
  }

  function renderHourlyScroller(hourly) {
    const scroller = document.getElementById('hourly-scroller');
    if (!scroller || !hourly) return;

    scroller.innerHTML = '';
    hourly.forEach(item => {
      const card = document.createElement('div');
      card.className = 'ww-hourly-card';
      card.innerHTML = `
        <div class="ww-hourly-time">${item.hour}</div>
        <div class="ww-hourly-icon"><i class="bi ${item.icon}"></i></div>
        <div class="ww-hourly-temp">${Math.round(item.temp)}°</div>
        <div class="ww-hourly-rain"><i class="bi bi-droplet-fill"></i> ${item.rain_prob}%</div>
      `;
      scroller.appendChild(card);
    });
  }

  // ==========================================================================
  // TAB 2: 7-DAY WEEKLY FORECAST
  // ==========================================================================

  async function fetchWeeklyForecast(forceRefresh = false) {
    try {
      let url = `/api/weather/weekly?city=${encodeURIComponent(state.city)}&mode=${state.mode}`;
      if (state.isGPS && state.lat !== null && state.lon !== null) {
        url += `&lat=${state.lat}&lon=${state.lon}`;
      }
      if (forceRefresh) {
        url += '&refresh=true';
      }
      const res = await fetch(url);
      const data = await res.json();
      if (!data || !data.forecast) return;

      const container = document.getElementById('weekly-cards-container');
      if (!container) return;

      container.innerHTML = '';
      data.forecast.forEach(day => {
        const card = document.createElement('div');
        card.className = 'ww-day-card';
        card.innerHTML = `
          <div>
            <div class="ww-day-name">${day.day_name}</div>
            <div class="ww-day-date">${day.date}</div>
            <div class="ww-day-icon"><i class="bi ${day.icon}"></i></div>
            <div class="ww-day-condition">${day.condition_name}</div>
          </div>
          <div>
            <div class="ww-temp-spread">
              <span class="ww-temp-high">${Math.round(day.high)}°</span>
              <span class="ww-temp-low">${Math.round(day.low)}°</span>
            </div>
            <div class="ww-rain-pill"><i class="bi bi-droplet-fill"></i> ${day.rain_prob}% Rain</div>
          </div>
        `;
        container.appendChild(card);
      });
    } catch (err) {
      console.error('Error fetching weekly forecast:', err);
    }
  }

  // ==========================================================================
  // TAB 3A: HISTORICAL TRENDS
  // ==========================================================================

  function setupTrendsControls() {
    document.querySelectorAll('.ww-period-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.ww-period-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.trendsPeriod = btn.getAttribute('data-period');
        fetchTrendsData();
      });
    });

    document.querySelectorAll('.ww-var-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        document.querySelectorAll('.ww-var-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        state.trendsVariable = btn.getAttribute('data-var');
        fetchTrendsData();
      });
    });
  }

  async function fetchTrendsData() {
    try {
      const url = `/api/weather/trends?city=${encodeURIComponent(state.city)}&period=${state.trendsPeriod}&variable=${state.trendsVariable}&mode=${state.mode}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!data || data.error) return;

      const avgEl = document.getElementById('stat-trend-mean');
      if (avgEl) avgEl.textContent = `${data.mean} ${data.unit}`;

      const maxEl = document.getElementById('stat-trend-max');
      if (maxEl) maxEl.textContent = `${data.max} ${data.unit}`;

      const minEl = document.getElementById('stat-trend-min');
      if (minEl) minEl.textContent = `${data.min} ${data.unit}`;

      const dirEl = document.getElementById('stat-trend-dir');
      if (dirEl) {
        dirEl.textContent = data.trend_direction.toUpperCase();
        dirEl.style.color = (data.trend_direction === 'warming' || data.trend_direction === 'increasing') ? '#d97706' : '#0284c7';
      }

      const narrativeEl = document.getElementById('trend-narrative');
      if (narrativeEl) narrativeEl.textContent = data.narrative;

      if (window.WeatherWiseCharts) {
        window.WeatherWiseCharts.renderTrendChart('trend-chart-container', data);
      }
    } catch (err) {
      console.error('Error fetching trends:', err);
    }
  }

  // ==========================================================================
  // TAB 3B: MONTHLY COMPARISONS
  // ==========================================================================

  async function fetchMonthlyData() {
    try {
      const url = `/api/weather/monthly?city=${encodeURIComponent(state.city)}&mode=${state.mode}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!data || data.error) return;

      if (window.WeatherWiseCharts) {
        window.WeatherWiseCharts.renderMonthlyChart('monthly-chart-container', data);
      }

      const tbody = document.getElementById('monthly-table-body');
      if (tbody && data.monthly_summary) {
        tbody.innerHTML = '';
        data.monthly_summary.forEach(row => {
          const tr = document.createElement('tr');
          tr.innerHTML = `
            <td class="fw-bold">${row.month}</td>
            <td>${row.avg_temp}°C</td>
            <td>${row.avg_rainfall} mm</td>
            <td>${row.avg_humidity}%</td>
            <td>${row.avg_wind} km/h</td>
          `;
          tbody.appendChild(tr);
        });
      }
    } catch (err) {
      console.error('Error fetching monthly comparisons:', err);
    }
  }

  // ==========================================================================
  // TAB 3C: SEASONAL PATTERNS
  // ==========================================================================

  async function fetchSeasonsData() {
    try {
      const url = `/api/weather/seasons?city=${encodeURIComponent(state.city)}&mode=${state.mode}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!data || data.error) return;

      const container = document.getElementById('seasons-cards-container');
      if (container && data.seasons) {
        container.innerHTML = '';
        data.seasons.forEach(s => {
          const col = document.createElement('div');
          col.className = 'col-md-3 col-sm-6';
          col.innerHTML = `
            <div class="p-3 bg-white bg-opacity-75 rounded-3 border text-center">
              <span class="badge bg-primary bg-opacity-10 text-primary mb-2">${s.season}</span>
              <div class="fw-bold" style="font-size:1.6rem; color:var(--text-main);">${s.mean_temp}°C</div>
              <div class="text-muted" style="font-size:0.75rem;">Avg Seasonal Temp</div>
              <hr class="my-2" style="opacity:0.1;">
              <div class="d-flex justify-content-between text-muted" style="font-size:0.8rem;">
                <span>Rain: <b>${s.total_rainfall} mm</b></span>
                <span>Wind: <b>${s.mean_wind} km/h</b></span>
              </div>
            </div>
          `;
          container.appendChild(col);
        });
      }

      const highlightsEl = document.getElementById('seasons-highlights');
      if (highlightsEl) {
        highlightsEl.innerHTML = `
          In <b>${data.city}</b>, the warmest season is <b>${data.warmest_season}</b>, while the coldest period is <b>${data.coldest_season}</b>.
          Precipitation peaks during <b>${data.wettest_season}</b>, with the lowest rainfall observed in <b>${data.driest_season}</b>.
          ${data.is_southern_hemisphere ? '<br><small class="text-info mt-1 d-block"><i class="bi bi-compass"></i> Southern Hemisphere seasonal alignment applied.</small>' : ''}
        `;
      }
    } catch (err) {
      console.error('Error fetching seasonal patterns:', err);
    }
  }

  // ==========================================================================
  // TAB 3D: UNUSUAL-WEATHER ANOMALY DETECTION
  // ==========================================================================

  function setupAnomalyControls() {
    const varSel = document.getElementById('anomaly-var-select');
    const methodSel = document.getElementById('anomaly-method-select');

    if (varSel) {
      varSel.addEventListener('change', (e) => {
        state.anomalyVariable = e.target.value;
        fetchAnomaliesData();
      });
    }

    if (methodSel) {
      methodSel.addEventListener('change', (e) => {
        state.anomalyMethod = e.target.value;
        fetchAnomaliesData();
      });
    }
  }

  async function fetchAnomaliesData() {
    try {
      const url = `/api/weather/anomalies?city=${encodeURIComponent(state.city)}&variable=${state.anomalyVariable}&method=${state.anomalyMethod}&mode=${state.mode}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!data || data.error) return;

      const countEl = document.getElementById('stat-anomaly-count');
      if (countEl) countEl.textContent = data.total_anomalies;

      const pctEl = document.getElementById('stat-anomaly-pct');
      if (pctEl) pctEl.textContent = `${data.anomaly_percentage}%`;

      const tbody = document.getElementById('anomalies-table-body');
      if (tbody && data.anomalies) {
        tbody.innerHTML = '';
        if (data.anomalies.length === 0) {
          tbody.innerHTML = '<tr><td colspan="5" class="text-center text-muted py-3">No extreme statistical outliers detected for this period.</td></tr>';
        } else {
          data.anomalies.forEach(a => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
              <td class="fw-bold">${a.date}</td>
              <td class="text-danger fw-bold">${a.value}</td>
              <td>${a.baseline}</td>
              <td><span class="badge ${a.diff > 0 ? 'bg-danger' : 'bg-info text-dark'}">${a.diff > 0 ? '+' : ''}${a.diff}</span></td>
              <td class="text-muted" style="font-size:0.85rem;">${a.explanation}</td>
            `;
            tbody.appendChild(tr);
          });
        }
      }
    } catch (err) {
      console.error('Error fetching anomalies:', err);
    }
  }

  // ==========================================================================
  // TAB 3E: FORECASTING ACCURACY EVALUATION
  // ==========================================================================

  async function fetchAccuracyData() {
    try {
      const url = `/api/weather/forecast-accuracy?city=${encodeURIComponent(state.city)}&variable=temperature&mode=${state.mode}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!data || data.error) return;

      const bMae = document.getElementById('stat-base-mae');
      if (bMae) bMae.textContent = `${data.baseline.mae}°C`;

      const bRmse = document.getElementById('stat-base-rmse');
      if (bRmse) bRmse.textContent = `${data.baseline.rmse}°C`;

      const esMae = document.getElementById('stat-es-mae');
      if (esMae) esMae.textContent = `${data.exponential_smoothing.mae}°C`;

      const esRmse = document.getElementById('stat-es-rmse');
      if (esRmse) esRmse.textContent = `${data.exponential_smoothing.rmse}°C`;

      const verdictEl = document.getElementById('accuracy-verdict');
      if (verdictEl) verdictEl.textContent = data.verdict;
    } catch (err) {
      console.error('Error fetching forecasting accuracy:', err);
    }
  }

  // ==========================================================================
  // TAB 4: FUTURE ANALYTICS & CLIMATE OUTLOOK
  // ==========================================================================

  async function fetchClimateData() {
    try {
      const url = `/api/weather/climate?city=${encodeURIComponent(state.city)}&mode=${state.mode}`;
      const res = await fetch(url);
      const data = await res.json();
      if (!data || data.error) return;

      const annualEl = document.getElementById('climate-annual-mean');
      if (annualEl) annualEl.textContent = `${data.annual_projected_mean}°C`;

      const warmestEl = document.getElementById('climate-warmest');
      if (warmestEl) warmestEl.textContent = data.warmest_month;

      const coolestEl = document.getElementById('climate-coolest');
      if (coolestEl) coolestEl.textContent = data.coolest_month;

      const wettestEl = document.getElementById('climate-wettest');
      if (wettestEl) wettestEl.textContent = data.wettest_month;

      const driestEl = document.getElementById('climate-driest');
      if (driestEl) driestEl.textContent = data.driest_month;

      if (window.WeatherWiseCharts) {
        window.WeatherWiseCharts.renderClimateChart('climate-chart-container', data);
      }
    } catch (err) {
      console.error('Error fetching climate data:', err);
    }
  }

  // ==========================================================================
  // TAB 5: PREPAREDNESS & LIFESTYLE GUIDE
  // ==========================================================================

  async function fetchPreparednessData() {
    try {
      let url = `/api/weather/preparedness?city=${encodeURIComponent(state.city)}&mode=${state.mode}`;
      if (state.isGPS && state.lat !== null && state.lon !== null) {
        url += `&lat=${state.lat}&lon=${state.lon}`;
      }
      const res = await fetch(url);
      const data = await res.json();
      if (!data || data.error) return;

      const gearList = document.getElementById('prep-gear-list');
      if (gearList && data.gear) {
        gearList.innerHTML = '';
        data.gear.forEach(item => {
          const card = document.createElement('div');
          card.className = 'ww-gear-card';
          card.innerHTML = `
            <div class="ww-gear-icon-box"><i class="bi ${item.icon}"></i></div>
            <div style="flex:1;">
              <div class="ww-gear-item-name">
                ${item.item}
                <span class="badge ${item.badge_class}" style="font-size:0.7rem; font-weight:600;">${item.priority}</span>
              </div>
              <div class="ww-gear-reason">${item.reason}</div>
            </div>
          `;
          gearList.appendChild(card);
        });
      }

      const attireTitle = document.getElementById('prep-attire-title');
      if (attireTitle && data.attire) attireTitle.textContent = data.attire.title;

      const attireDesc = document.getElementById('prep-attire-desc');
      if (attireDesc && data.attire) attireDesc.textContent = data.attire.description;

      const hydrationVal = document.getElementById('prep-hydration-val');
      if (hydrationVal) hydrationVal.textContent = `${data.hydration_target_litres} Litres`;

      const activityScore = document.getElementById('prep-activity-score');
      if (activityScore) activityScore.textContent = data.activity_score;

      const activityVerdict = document.getElementById('prep-activity-verdict');
      if (activityVerdict) activityVerdict.textContent = data.activity_verdict;

      const commuteLevel = document.getElementById('prep-commute-level');
      if (commuteLevel && data.commute) {
        commuteLevel.textContent = data.commute.level;
        commuteLevel.className = `badge ${data.commute.badge_class}`;
      }

      const commuteAlert = document.getElementById('prep-commute-alert');
      if (commuteAlert && data.commute) commuteAlert.textContent = data.commute.alert;

    } catch (err) {
      console.error('Error fetching preparedness data:', err);
    }
  }

  // ==========================================================================
  // SUB-TAB NAVIGATION (INSIDE TAB 3)
  // ==========================================================================

  function setupSubtabListeners() {
    document.querySelectorAll('.ww-subtab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetSubtab = btn.getAttribute('data-subtab');
        if (!targetSubtab) return;

        state.activeSubtab = targetSubtab;

        document.querySelectorAll('.ww-subtab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.ww-subtab-pane').forEach(pane => {
          if (pane.id === targetSubtab) {
            pane.style.display = 'block';
          } else {
            pane.style.display = 'none';
          }
        });

        // Resize charts on subtab switch
        if (targetSubtab === 'subtab-trends' && window.Plotly) {
          setTimeout(() => {
            const chart = document.getElementById('trend-chart-container');
            if (chart) Plotly.Plots.resize(chart);
          }, 50);
        } else if (targetSubtab === 'subtab-monthly' && window.Plotly) {
          setTimeout(() => {
            const chart = document.getElementById('monthly-chart-container');
            if (chart) Plotly.Plots.resize(chart);
          }, 50);
        }
      });
    });
  }

  // ==========================================================================
  // TAB NAVIGATION
  // ==========================================================================

  function setupTabListeners() {
    document.querySelectorAll('.ww-tab-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const targetTab = btn.getAttribute('data-tab');
        if (!targetTab) return;

        state.activeTab = targetTab;

        document.querySelectorAll('.ww-tab-btn').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');

        document.querySelectorAll('.ww-tab-pane').forEach(pane => {
          if (pane.id === targetTab) {
            pane.style.display = 'block';
          } else {
            pane.style.display = 'none';
          }
        });

        if (targetTab === 'tab-trends' && window.Plotly) {
          setTimeout(() => {
            const chart = document.getElementById('trend-chart-container');
            if (chart) Plotly.Plots.resize(chart);
          }, 50);
        } else if (targetTab === 'tab-climate' && window.Plotly) {
          setTimeout(() => {
            const chart = document.getElementById('climate-chart-container');
            if (chart) Plotly.Plots.resize(chart);
          }, 50);
        }
      });
    });
  }

  document.addEventListener('DOMContentLoaded', init);
})();
