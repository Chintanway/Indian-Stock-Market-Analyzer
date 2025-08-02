// Indian Stock Market Analyzer - Frontend JavaScript

document.addEventListener('DOMContentLoaded', function() {
    const analyzeForm = document.getElementById('analyzeForm');
    const symbolInput = document.getElementById('symbol');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const loading = document.getElementById('loading');
    const error = document.getElementById('error');
    const results = document.getElementById('results');
    const quickSymbolBtns = document.querySelectorAll('.quick-symbol');

    // Quick symbol selection
    quickSymbolBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            const symbol = this.getAttribute('data-symbol');
            symbolInput.value = symbol;
            analyzeForm.dispatchEvent(new Event('submit'));
        });
    });

    // Form submission
    analyzeForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const symbol = symbolInput.value.trim().toUpperCase();
        if (!symbol) {
            showError('Please enter a stock symbol');
            return;
        }

        analyzeStock(symbol);
    });

    function analyzeStock(symbol) {
        // Show loading state
        showLoading();
        hideError();
        hideResults();

        // Prepare form data
        const formData = new FormData();
        formData.append('symbol', symbol);

        // Make API call
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            hideLoading();
            
            if (data.success) {
                displayResults(data.data);
            } else {
                showError(data.error || 'An error occurred while analyzing the stock');
            }
        })
        .catch(err => {
            hideLoading();
            showError('Network error: ' + err.message);
        });
    }

    function displayResults(data) {
        // Update stock title
        document.getElementById('stockTitle').innerHTML = 
            `<i class="fas fa-chart-line me-2"></i>${data.symbol} Analysis`;

        // Update price information
        document.getElementById('currentPrice').textContent = `₹${data.current_price}`;
        document.getElementById('predictedPrice').textContent = `₹${data.predicted_price}`;
        document.getElementById('targetPrice').textContent = `₹${data.target_price}`;
        document.getElementById('stopLoss').textContent = `₹${data.stop_loss}`;

        // Update signal
        const signalElement = document.getElementById('signal');
        const signal = data.signal;
        signalElement.textContent = signal;
        signalElement.className = 'badge fs-6 ' + getSignalClass(signal);

        // Update prediction change
        const changeElement = document.getElementById('predictionChange');
        const change = data.prediction_change;
        changeElement.textContent = `${change > 0 ? '+' : ''}${change}%`;
        changeElement.className = 'h5 ' + (change > 0 ? 'price-up' : change < 0 ? 'price-down' : 'price-neutral');

        // Update data timestamp
        if (data.data_timestamp) {
            document.getElementById('dataTimestamp').textContent = data.data_timestamp;
        }

        // Update investment recommendation
        if (data.investment_recommendation) {
            updateInvestmentRecommendation(data.investment_recommendation);
        }

        // Update market analysis
        if (data.sentiment_analysis && data.risk_analysis) {
            updateMarketAnalysis(data.sentiment_analysis, data.risk_analysis);
        }

        // Update technical indicators
        updateIndicators(data.indicators, data.current_price);

        // Update AI model performance
        updateModelInfo(data.model_stats);

        // Display chart
        displayChart(data.chart);

        // Load analysis history
        loadAnalysisHistory(data.symbol);

        // Show results
        showResults();
    }

    function updateIndicators(indicators, currentPrice) {
        // RSI
        const rsi = indicators.rsi;
        document.getElementById('rsi').textContent = rsi.toFixed(2);
        document.getElementById('rsiStatus').textContent = getRSIStatus(rsi);
        document.getElementById('rsiStatus').className = 'indicator-status ' + getRSIStatusClass(rsi);

        // EMA 5
        const ema5 = indicators.ema_5;
        document.getElementById('ema5').textContent = `₹${ema5.toFixed(2)}`;
        document.getElementById('ema5Status').textContent = getEMAStatus(currentPrice, ema5);
        document.getElementById('ema5Status').className = 'indicator-status ' + getEMAStatusClass(currentPrice, ema5);

        // MACD
        const macd = indicators.macd;
        document.getElementById('macd').textContent = macd.toFixed(4);
        document.getElementById('macdStatus').textContent = getMACDStatus(macd);
        document.getElementById('macdStatus').className = 'indicator-status ' + getMACDStatusClass(macd);

        // Other indicators
        document.getElementById('sma20').textContent = `₹${indicators.sma_20.toFixed(2)}`;
        document.getElementById('bbUpper').textContent = `₹${indicators.bb_upper.toFixed(2)}`;
        document.getElementById('bbLower').textContent = `₹${indicators.bb_lower.toFixed(2)}`;
    }

    function updateModelInfo(modelStats) {
        document.getElementById('trainAccuracy').textContent = `${modelStats.train_accuracy}%`;
        document.getElementById('testAccuracy').textContent = `${modelStats.test_accuracy}%`;
        document.getElementById('modelInfo').style.display = 'block';
    }

    function updateInvestmentRecommendation(rec) {
        document.getElementById('investmentRecommendation').style.display = 'block';
        
        const recBadge = document.getElementById('recommendationBadge');
        const recIcon = document.getElementById('recommendationIcon');
        const recContainer = document.getElementById('recommendationContainer');
        
        recBadge.textContent = rec.recommendation;
        recBadge.className = `badge fs-4 px-3 py-2 bg-${rec.action_color}`;
        
        // Set appropriate icon and container styling based on recommendation
        switch(rec.recommendation) {
            case 'STRONG BUY':
                recIcon.textContent = '🚀';
                recContainer.style.background = 'linear-gradient(135deg, #d4edda, #c3e6cb)';
                break;
            case 'BUY':
                recIcon.textContent = '📈';
                recContainer.style.background = 'linear-gradient(135deg, #d4edda, #c3e6cb)';
                break;
            case 'HOLD':
                recIcon.textContent = '⏸️';
                recContainer.style.background = 'linear-gradient(135deg, #fff3cd, #fce57a)';
                break;
            case 'SELL':
                recIcon.textContent = '📉';
                recContainer.style.background = 'linear-gradient(135deg, #f8d7da, #f1b0b7)';
                break;
            case 'STRONG SELL':
                recIcon.textContent = '🔻';
                recContainer.style.background = 'linear-gradient(135deg, #f8d7da, #f1b0b7)';
                break;
            default:
                recIcon.textContent = '📊';
                recContainer.style.background = 'linear-gradient(135deg, #f8f9fa, #e9ecef)';
        }
        
        document.getElementById('confidenceLevel').textContent = rec.confidence_level;
        document.getElementById('riskLevel').textContent = rec.risk_level;
        document.getElementById('positionSize').textContent = rec.position_size;
        document.getElementById('investmentHorizon').textContent = rec.investment_horizon;
        document.getElementById('riskRewardRatio').textContent = `1:${rec.risk_reward_ratio}`;
        
        // Update key reasons
        const reasonsList = document.getElementById('keyReasons');
        reasonsList.innerHTML = rec.key_reasons.map(reason => `<li>${reason}</li>`).join('');
    }

    function updateMarketAnalysis(sentiment, risk) {
        document.getElementById('marketAnalysis').style.display = 'block';
        
        document.getElementById('momentum1w').textContent = `${sentiment.momentum_1w > 0 ? '+' : ''}${sentiment.momentum_1w.toFixed(1)}%`;
        document.getElementById('momentum1m').textContent = `${sentiment.momentum_1m > 0 ? '+' : ''}${sentiment.momentum_1m.toFixed(1)}%`;
        document.getElementById('volatility').textContent = `${risk.volatility.toFixed(1)}%`;
        document.getElementById('sector').textContent = sentiment.sector;
        document.getElementById('volumeActivity').textContent = `${sentiment.volume_surge.toFixed(0)}% of average`;
    }

    function displayChart(chartData) {
        const chartConfig = JSON.parse(chartData);
        Plotly.newPlot('priceChart', chartConfig.data, chartConfig.layout, {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d', 'autoScale2d'],
            displaylogo: false
        });
    }

    // Helper functions for indicator status
    function getRSIStatus(rsi) {
        if (rsi < 30) return 'Oversold';
        if (rsi > 70) return 'Overbought';
        return 'Neutral';
    }

    function getRSIStatusClass(rsi) {
        if (rsi < 30) return 'bullish';
        if (rsi > 70) return 'bearish';
        return 'neutral';
    }

    function getEMAStatus(currentPrice, ema) {
        if (currentPrice > ema) return 'Above EMA';
        if (currentPrice < ema) return 'Below EMA';
        return 'At EMA';
    }

    function getEMAStatusClass(currentPrice, ema) {
        if (currentPrice > ema) return 'bullish';
        if (currentPrice < ema) return 'bearish';
        return 'neutral';
    }

    function getMACDStatus(macd) {
        if (macd > 0) return 'Bullish';
        if (macd < 0) return 'Bearish';
        return 'Neutral';
    }

    function getMACDStatusClass(macd) {
        if (macd > 0) return 'bullish';
        if (macd < 0) return 'bearish';
        return 'neutral';
    }

    function getSignalClass(signal) {
        switch (signal) {
            case 'BUY': return 'bg-success';
            case 'SELL': return 'bg-danger';
            case 'CAUTION': return 'bg-warning';
            case 'NEUTRAL': return 'bg-secondary';
            default: return 'bg-secondary';
        }
    }

    // UI state management functions
    function showLoading() {
        loading.style.display = 'block';
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = '<i class="fas fa-spinner fa-spin me-2"></i>Analyzing...';
    }

    function hideLoading() {
        loading.style.display = 'none';
        analyzeBtn.disabled = false;
        analyzeBtn.innerHTML = '<i class="fas fa-chart-bar me-2"></i>Analyze Stock';
    }

    function showError(message) {
        error.textContent = message;
        error.style.display = 'block';
    }

    function hideError() {
        error.style.display = 'none';
    }

    function showResults() {
        results.style.display = 'block';
        // Smooth scroll to results
        results.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    function hideResults() {
        results.style.display = 'none';
        document.getElementById('modelInfo').style.display = 'none';
    }

    // Input validation and formatting
    symbolInput.addEventListener('input', function() {
        // Allow only letters and limit length
        this.value = this.value.toUpperCase().replace(/[^A-Z]/g, '').substring(0, 20);
        
        // Show suggestion for common long symbols
        const suggestions = {
            'ADANIGR': 'ADANIGREEN',
            'ADANIGREEN': 'ADANIGREEN ✓',
            'RELIANCEI': 'RELIANCE',
            'RELIANCE': 'RELIANCE ✓',
            'TATACON': 'TCS',
            'TCS': 'TCS ✓',
            'INFOSYS': 'INFY',
            'INFY': 'INFY ✓',
            'JSW': 'JSWSTEEL',
            'JSWSTEEL': 'JSWSTEEL ✓',
            'MARUTI': 'MARUTI ✓',
            'MARUTISU': 'MARUTI',
            'LARSEN': 'LT',
            'LT': 'LT ✓'
        };
        
        // Remove any existing suggestions
        const existingSuggestion = document.getElementById('symbolSuggestion');
        if (existingSuggestion) {
            existingSuggestion.remove();
        }
        
        // Show suggestion if applicable
        const value = this.value;
        if (suggestions[value]) {
            const suggestion = document.createElement('small');
            suggestion.id = 'symbolSuggestion';
            suggestion.className = 'text-muted';
            suggestion.textContent = `Suggestion: ${suggestions[value]}`;
            this.parentNode.appendChild(suggestion);
        }
    });

    // Auto-focus on symbol input
    symbolInput.focus();

    // Load popular stocks from database
    loadPopularStocks();
    
    // Load market overview functions
    window.loadMarketOverview = function() {
        const marketGrid = document.getElementById('marketGrid');
        if (!marketGrid) return;
        
        marketGrid.innerHTML = '<div class="col-12 text-center"><i class="fas fa-spinner fa-spin fa-2x text-primary"></i><p class="mt-2">Loading market overview...</p></div>';
        
        fetch('/api/market-overview')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayMarketOverview(data.data);
                } else {
                    marketGrid.innerHTML = '<div class="col-12 text-center text-danger"><i class="fas fa-exclamation-triangle fa-2x mb-2"></i><p>Error loading market data</p></div>';
                }
            })
            .catch(error => {
                console.error('Market overview error:', error);
                marketGrid.innerHTML = '<div class="col-12 text-center text-danger"><i class="fas fa-exclamation-triangle fa-2x mb-2"></i><p>Error loading market data</p></div>';
            });
    };

    function displayMarketOverview(recommendations) {
        const marketGrid = document.getElementById('marketGrid');
        if (!marketGrid) return;
        
        if (!recommendations || recommendations.length === 0) {
            marketGrid.innerHTML = '<div class="col-12 text-center text-muted"><p>No market data available</p></div>';
            return;
        }

        let html = '';
        recommendations.forEach(stock => {
            const iconMap = {
                'STRONG BUY': '🚀',
                'BUY': '📈',
                'HOLD': '⏸️',
                'SELL': '📉',
                'STRONG SELL': '🔻'
            };
            
            const colorMap = {
                'STRONG BUY': 'success',
                'BUY': 'primary', 
                'HOLD': 'warning',
                'SELL': 'danger',
                'STRONG SELL': 'dark'
            };

            html += `
                <div class="col-lg-4 col-md-6 mb-3">
                    <div class="card h-100 shadow-sm border-0" style="transition: transform 0.2s;">
                        <div class="card-body p-3">
                            <div class="d-flex justify-content-between align-items-start mb-2">
                                <div>
                                    <h6 class="card-title mb-1 fw-bold">${stock.symbol}</h6>
                                    <small class="text-muted">${stock.sector}</small>
                                </div>
                                <div class="text-end">
                                    <div class="fs-4">${iconMap[stock.recommendation] || '📊'}</div>
                                </div>
                            </div>
                            
                            <div class="mb-2">
                                <div class="d-flex justify-content-between">
                                    <span class="fw-bold">₹${stock.current_price}</span>
                                    <span class="badge bg-${colorMap[stock.recommendation]} px-2">${stock.recommendation}</span>
                                </div>
                            </div>
                            
                            <div class="row text-center mb-2">
                                <div class="col-4">
                                    <small class="text-muted d-block">Signal</small>
                                    <small class="fw-bold">${stock.signal}</small>
                                </div>
                                <div class="col-4">
                                    <small class="text-muted d-block">Risk</small>
                                    <small class="fw-bold">${stock.risk_level}</small>
                                </div>
                                <div class="col-4">
                                    <small class="text-muted d-block">Return</small>
                                    <small class="fw-bold ${stock.potential_return > 0 ? 'text-success' : 'text-danger'}">${stock.potential_return > 0 ? '+' : ''}${stock.potential_return}%</small>
                                </div>
                            </div>
                            
                            <div class="mt-2">
                                <small class="text-muted">${stock.key_reason}</small>
                            </div>
                            
                            <button class="btn btn-outline-primary btn-sm w-100 mt-2" onclick="analyzeStock('${stock.symbol}')">
                                <i class="fas fa-chart-line me-1"></i>Detailed Analysis
                            </button>
                        </div>
                    </div>
                </div>
            `;
        });

        marketGrid.innerHTML = html;
    }

    window.analyzeStock = function(symbol) {
        document.getElementById('symbol').value = symbol;
        document.getElementById('analyzeForm').dispatchEvent(new Event('submit'));
    };
    
    // Auto-load market overview
    setTimeout(() => {
        if (typeof loadMarketOverview === 'function') {
            loadMarketOverview();
        }
    }, 1500);

    // Handle Enter key in symbol input
    symbolInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            analyzeForm.dispatchEvent(new Event('submit'));
        }
    });

    // Load popular stocks from database
    function loadPopularStocks() {
        fetch('/api/popular-stocks')
            .then(response => response.json())
            .then(data => {
                if (data.success && data.data.length > 0) {
                    const container = document.getElementById('popularStocks');
                    // Clear existing default buttons
                    container.innerHTML = '';
                    
                    // Add popular stocks from database
                    data.data.forEach(stock => {
                        const button = document.createElement('button');
                        button.className = 'btn btn-outline-secondary btn-sm quick-symbol me-1 mb-1';
                        button.setAttribute('data-symbol', stock.symbol);
                        button.textContent = `${stock.symbol} (${stock.count})`;
                        button.addEventListener('click', function() {
                            const symbol = this.getAttribute('data-symbol');
                            symbolInput.value = symbol;
                            analyzeForm.dispatchEvent(new Event('submit'));
                        });
                        container.appendChild(button);
                    });
                }
            })
            .catch(err => console.log('Could not load popular stocks'));
    }

    // Load analysis history for a symbol
    function loadAnalysisHistory(symbol) {
        fetch(`/api/analysis-history/${symbol}`)
            .then(response => response.json())
            .then(data => {
                if (data.success && data.data.length > 0) {
                    const tbody = document.getElementById('historyTable');
                    tbody.innerHTML = '';
                    
                    data.data.forEach(analysis => {
                        const row = document.createElement('tr');
                        const changeClass = analysis.prediction_change > 0 ? 'text-success' : 
                                          analysis.prediction_change < 0 ? 'text-danger' : 'text-muted';
                        const signalClass = analysis.signal === 'BUY' ? 'text-success' :
                                          analysis.signal === 'SELL' ? 'text-danger' : 'text-warning';
                        
                        row.innerHTML = `
                            <td><small>${analysis.date}</small></td>
                            <td>₹${analysis.current_price}</td>
                            <td>₹${analysis.predicted_price}</td>
                            <td class="${changeClass}">${analysis.prediction_change > 0 ? '+' : ''}${analysis.prediction_change}%</td>
                            <td><span class="badge badge-sm ${signalClass}">${analysis.signal}</span></td>
                        `;
                        tbody.appendChild(row);
                    });
                    
                    document.getElementById('historyCard').style.display = 'block';
                }
            })
            .catch(err => console.log('Could not load analysis history'));
    }

    // IPO Tracker functionality
    const refreshIPOsBtn = document.getElementById('refreshIPOs');
    const viewCalendarBtn = document.getElementById('viewCalendar');
    const ipoLoading = document.getElementById('ipoLoading');
    const ipoResults = document.getElementById('ipoResults');
    const ipoSummary = document.getElementById('ipoSummary');
    const ipoList = document.getElementById('ipoList');
    const noIPOs = document.getElementById('noIPOs');
    const ipoLastUpdated = document.getElementById('ipoLastUpdated');
    
    // IPO Tab click handler
    const ipoTab = document.getElementById('ipo-tracker-tab');
    if (ipoTab) {
        ipoTab.addEventListener('click', function() {
            // Load IPOs when tab is clicked
            setTimeout(() => {
                loadUpcomingIPOs();
            }, 100);
        });
    }
    
    // Refresh IPOs button
    if (refreshIPOsBtn) {
        refreshIPOsBtn.addEventListener('click', function() {
            loadUpcomingIPOs();
        });
    }
    
    // View Calendar button
    if (viewCalendarBtn) {
        viewCalendarBtn.addEventListener('click', function() {
            loadIPOCalendar(30);
        });
    }
    
    function loadUpcomingIPOs() {
        showIPOLoading();
        
        fetch('/upcoming-ipos')
            .then(response => response.json())
            .then(data => {
                hideIPOLoading();
                if (data.success) {
                    displayIPOs(data);
                } else {
                    showIPOError(data.error || 'Failed to load IPO data');
                }
            })
            .catch(error => {
                hideIPOLoading();
                console.error('Error loading IPOs:', error);
                showIPOError('Network error while loading IPO data');
            });
    }
    
    function loadIPOCalendar(days) {
        showIPOLoading();
        
        fetch(`/ipo-calendar?days=${days}`)
            .then(response => response.json())
            .then(data => {
                hideIPOLoading();
                if (data.success) {
                    displayIPOs(data);
                } else {
                    showIPOError(data.error || 'Failed to load IPO calendar');
                }
            })
            .catch(error => {
                hideIPOLoading();
                console.error('Error loading IPO calendar:', error);
                showIPOError('Network error while loading IPO calendar');
            });
    }
    
    function displayIPOs(data) {
        // Update summary counts
        if (ipoSummary) {
            document.getElementById('openIPOCount').textContent = data.open_count || 0;
            document.getElementById('upcomingIPOCount').textContent = data.upcoming_count || 0;
            document.getElementById('totalIPOCount').textContent = data.total_count || 0;
            ipoSummary.style.display = data.total_count > 0 ? 'flex' : 'none';
        }
        
        // Update last updated time
        if (ipoLastUpdated && data.last_updated) {
            ipoLastUpdated.textContent = `Last updated: ${data.last_updated}`;
        }
        
        // Display IPO list
        if (data.total_count > 0) {
            displayIPOList(data.ipos);
            noIPOs.style.display = 'none';
        } else {
            ipoList.innerHTML = '';
            noIPOs.style.display = 'block';
        }
    }
    
    function displayIPOList(ipos) {
        if (!ipoList) return;
        
        let html = '';
        
        ipos.forEach(ipo => {
            const statusClass = ipo.status === 'Open' ? 'success' : 'warning';
            const statusIcon = ipo.status === 'Open' ? 'fa-check-circle' : 'fa-clock';
            
            html += `
                <div class="card mb-3 border-${statusClass}">
                    <div class="card-header bg-${statusClass} text-white">
                        <div class="d-flex justify-content-between align-items-center">
                            <h6 class="mb-0">
                                <i class="fas ${statusIcon} me-2"></i>
                                ${ipo.company_name}
                            </h6>
                            <span class="badge bg-light text-dark">${ipo.status}</span>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3">
                                <strong>Issue Size:</strong><br>
                                <span class="text-primary">${ipo.issue_size}</span>
                            </div>
                            <div class="col-md-3">
                                <strong>Price Band:</strong><br>
                                <span class="text-info">${ipo.price_band}</span>
                            </div>
                            <div class="col-md-3">
                                <strong>Closing Date:</strong><br>
                                <span class="text-danger">${ipo.closing_date}</span>
                                <br><small class="text-muted">${ipo.days_to_close}</small>
                            </div>
                            <div class="col-md-3">
                                <strong>Lot Size:</strong><br>
                                <span>${ipo.lot_size}</span>
                            </div>
                        </div>
                        <hr>
                        <div class="row">
                            <div class="col-md-4">
                                <strong>Opening Date:</strong> ${ipo.opening_date}
                            </div>
                            <div class="col-md-4">
                                <strong>Listing Date:</strong> ${ipo.listing_date}
                            </div>
                            <div class="col-md-4">
                                <strong>Category:</strong> ${ipo.category}
                            </div>
                        </div>
                        <div class="row mt-2">
                            <div class="col-md-6">
                                <strong>Lead Managers:</strong><br>
                                <small class="text-muted">${ipo.lead_managers}</small>
                            </div>
                            <div class="col-md-6">
                                <strong>Registrar:</strong><br>
                                <small class="text-muted">${ipo.registrar}</small>
                            </div>
                        </div>
                    </div>
                </div>
            `;
        });
        
        ipoList.innerHTML = html;
    }
    
    function showIPOLoading() {
        if (ipoLoading) ipoLoading.style.display = 'block';
        if (ipoResults) ipoResults.style.display = 'none';
    }
    
    function hideIPOLoading() {
        if (ipoLoading) ipoLoading.style.display = 'none';
        if (ipoResults) ipoResults.style.display = 'block';
    }
    
    function showIPOError(message) {
        if (ipoList) {
            ipoList.innerHTML = `
                <div class="alert alert-danger" role="alert">
                    <i class="fas fa-exclamation-triangle me-2"></i>
                    ${message}
                </div>
            `;
        }
        if (noIPOs) noIPOs.style.display = 'none';
        if (ipoSummary) ipoSummary.style.display = 'none';
    }

    // Volume Tracker functionality
    const refreshVolumeBtn = document.getElementById('refreshVolume');
    const volumeLoading = document.getElementById('volumeLoading');
    const volumeError = document.getElementById('volumeError');
    const volumeErrorText = document.getElementById('volumeErrorText');
    const volumeLastUpdated = document.getElementById('volumeLastUpdated');

    // Initialize volume tracker when tab is shown
    document.getElementById('volume-tracker-tab').addEventListener('shown.bs.tab', function() {
        loadVolumeData();
    });

    // Refresh button click handler
    if (refreshVolumeBtn) {
        refreshVolumeBtn.addEventListener('click', function(e) {
            e.preventDefault();
            loadVolumeData();
        });
    }

    // Load volume data from API
    async function loadVolumeData() {
        try {
            showVolumeLoading();
            hideVolumeError();
            
            console.log('🔍 [DEBUG] Fetching volume data from /api/volume/top...');
            const response = await fetch('/api/volume/top', {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json'
                },
                credentials: 'same-origin'
            });
            
            if (!response.ok) {
                const errorData = await response.json().catch(() => ({}));
                throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            console.log('📊 [DEBUG] Volume data response:', JSON.stringify(result, null, 2));
            
            if (result.success && result.data) {
                console.log('✅ [DEBUG] Data received successfully, updating tables...');
                updateVolumeTables(result.data);
                updateLastUpdated();
            } else {
                const errorMsg = result.error || 'Failed to load volume data';
                console.error('❌ [DEBUG] Error in volume data:', errorMsg);
                throw new Error(errorMsg);
            }
        } catch (error) {
            console.error('Error loading volume data:', error);
            showVolumeError(error.message);
        } finally {
            hideVolumeLoading();
        }
    }

    // Update volume tables with data
    function updateVolumeTables(data) {
        if (!data) {
            console.error('❌ [DEBUG] No data provided to updateVolumeTables');
            return;
        }
        
        console.log('🔄 [DEBUG] Updating tables with data:', data);
        
        // Update most bought stocks
        const mostBoughtBody = document.getElementById('mostBoughtBody');
        if (data.most_bought && Array.isArray(data.most_bought) && data.most_bought.length > 0) {
            console.log('🔼 [DEBUG] Most bought stocks:', data.most_bought);
            mostBoughtBody.innerHTML = data.most_bought.map(stock => {
                console.log(`📝 [DEBUG] Processing bought stock:`, stock);
                return createVolumeTableRow(stock, 'bought');
            }).join('');
        } else {
            console.warn('⚠️ [DEBUG] No most bought stocks data in response');
            mostBoughtBody.innerHTML = '<tr><td colspan="6" class="text-center">No active buying data available</td></tr>';
        }
        
        // Update most sold stocks
        const mostSoldBody = document.getElementById('mostSoldBody');
        if (data.most_sold && Array.isArray(data.most_sold) && data.most_sold.length > 0) {
            console.log('🔽 [DEBUG] Most sold stocks:', data.most_sold);
            mostSoldBody.innerHTML = data.most_sold.map(stock => {
                console.log(`📝 [DEBUG] Processing sold stock:`, stock);
                return createVolumeTableRow(stock, 'sold');
            }).join('');
        } else {
            console.warn('⚠️ [DEBUG] No most sold stocks data in response');
            mostSoldBody.innerHTML = '<tr><td colspan="6" class="text-center">No active selling data available</td></tr>';
        }
        
        // Add event listeners to analyze buttons
        document.querySelectorAll('.analyze-volume-stock').forEach(btn => {
            btn.addEventListener('click', function() {
                const symbol = this.getAttribute('data-symbol');
                if (symbol) {
                    // Switch to stock analysis tab and trigger analysis
                    const stockTab = new bootstrap.Tab(document.getElementById('stock-analysis-tab'));
                    stockTab.show();
                    analyzeStock(symbol);
                }
            });
        });
    }

    // Create a table row for volume data
    function createVolumeTableRow(stock, type) {
        if (!stock) return '';
        
        const priceChange = stock.price_change || 0;
        const volumeChange = stock.volume_change || 0;
        const priceChangeClass = priceChange > 0 ? 'text-success' : priceChange < 0 ? 'text-danger' : '';
        const volumeChangeClass = volumeChange > 0 ? 'text-success' : volumeChange < 0 ? 'text-danger' : '';
        const priceChangeIcon = priceChange > 0 ? '▲' : priceChange < 0 ? '▼' : '';
        const volumeChangeIcon = volumeChange > 0 ? '▲' : volumeChange < 0 ? '▼' : '';
        
        return `
            <tr>
                <td><strong>${stock.symbol}</strong></td>
                <td>₹${stock.current_price?.toFixed(2) || 'N/A'}</td>
                <td class="${priceChangeClass}">
                    ${priceChangeIcon} ${Math.abs(priceChange).toFixed(2)}%
                </td>
                <td>${stock.volume ? stock.volume.toLocaleString() : 'N/A'}</td>
                <td class="${volumeChangeClass}">
                    ${volumeChangeIcon} ${Math.abs(volumeChange).toFixed(2)}%
                </td>
                <td>
                    <button class="btn btn-sm btn-outline-primary analyze-volume-stock" data-symbol="${stock.symbol}">
                        <i class="fas fa-search"></i> Analyze
                    </button>
                </td>
            </tr>
        `;
    }

    // Update last updated timestamp
    function updateLastUpdated() {
        const now = new Date();
        volumeLastUpdated.textContent = now.toLocaleString();
    }

    // Show/hide loading state
    function showVolumeLoading() {
        if (volumeLoading) volumeLoading.style.display = 'block';
    }

    function hideVolumeLoading() {
        if (volumeLoading) volumeLoading.style.display = 'none';
    }

    // Show/hide error message
    function showVolumeError(message) {
        if (volumeError && volumeErrorText) {
            volumeErrorText.textContent = message;
            volumeError.style.display = 'block';
        }
    }

    function hideVolumeError() {
        if (volumeError) volumeError.style.display = 'none';
    }

    // Add some visual feedback for better UX
    document.querySelectorAll('.card').forEach(card => {
        card.addEventListener('mouseenter', function() {
            this.style.transform = 'translateY(-2px)';
            this.style.boxShadow = '0 10px 20px rgba(0,0,0,0.1)';
        });
        card.addEventListener('mouseleave', function() {
            this.style.transform = '';
            this.style.boxShadow = '';
        });
    });
});