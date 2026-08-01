// Global Chart Instances
let modelChartInstance = null;
let shapChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    fetchHealthStatus();
    fetchMetricsAndCharts();
});

// Tab Switching Logic
function switchTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => btn.classList.remove("active"));
    document.querySelectorAll(".tab-content").forEach(content => content.classList.remove("active"));
    
    document.getElementById(tabId).classList.add("active");
    
    // Find active tab button
    const activeBtn = Array.from(document.querySelectorAll(".tab-btn")).find(btn => 
        btn.getAttribute("onclick").includes(tabId)
    );
    if (activeBtn) activeBtn.classList.add("active");
}

// Fetch System Health Status
async function fetchHealthStatus() {
    try {
        const res = await fetch("/health");
        const data = await res.json();
        const badge = document.getElementById("system-status");
        if (data.model_loaded) {
            badge.innerHTML = `<span class="pulse-dot"></span> System Live: ${data.model_name}`;
        } else {
            badge.style.borderColor = "rgba(244, 63, 94, 0.4)";
            badge.style.color = "#f43f5e";
            badge.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> Model Not Trained`;
        }
    } catch (err) {
        console.warn("Health check error:", err);
    }
}

// Single Applicant Prediction Form Handler
async function handlePrediction(event) {
    event.preventDefault();
    const btn = document.getElementById("predict-btn");
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Evaluating Model...`;
    btn.disabled = true;

    const payload = {
        age: parseInt(document.getElementById("age").value),
        annual_income: parseFloat(document.getElementById("annual_income").value),
        credit_score: parseInt(document.getElementById("credit_score").value),
        debt_to_income_ratio: parseFloat(document.getElementById("debt_to_income_ratio").value),
        credit_utilization_rate: parseFloat(document.getElementById("credit_utilization_rate").value),
        payment_history_score: parseFloat(document.getElementById("payment_history_score").value),
        loan_amount: parseFloat(document.getElementById("loan_amount").value),
        employment_length_years: parseInt(document.getElementById("employment_length_years").value),
        revolving_balance: parseFloat(document.getElementById("revolving_balance").value),
        num_credit_lines: parseInt(document.getElementById("num_credit_lines").value),
        home_ownership: document.getElementById("home_ownership").value,
        loan_intent: document.getElementById("loan_intent").value
    };

    try {
        const res = await fetch("/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!res.ok) {
            const errData = await res.json();
            alert(`Prediction Error: ${errData.detail || "Server error"}`);
            return;
        }

        const data = await res.json();
        updateResultGauge(data);
    } catch (err) {
        alert(`Connection Error: ${err.message}`);
    } finally {
        btn.innerHTML = `<i class="fa-solid fa-bolt"></i> Evaluate Default Risk`;
        btn.disabled = false;
    }
}

// Update Gauge and Results Display
function updateResultGauge(data) {
    const riskPercent = data.risk_score_percentage;
    const riskDisplay = document.getElementById("risk-score-display");
    const gaugeCircle = document.getElementById("gauge-circle");
    const badge = document.getElementById("risk-level-badge");
    const recText = document.getElementById("recommendation-text");
    const factorsList = document.getElementById("risk-factors-list");

    // Animate Gauge Number
    riskDisplay.innerText = `${riskPercent}%`;

    // Determine Gauge Color Palette
    let color = "#10b981"; // Emerald green for Low Risk
    if (data.risk_level === "MEDIUM RISK") color = "#f59e0b"; // Amber for Medium Risk
    if (data.risk_level === "HIGH RISK") color = "#f43f5e"; // Rose red for High Risk

    const deg = (riskPercent / 100) * 360;
    gaugeCircle.style.background = `conic-gradient(${color} ${deg}deg, rgba(255, 255, 255, 0.08) ${deg}deg)`;

    // Update Recommendation Text
    badge.innerText = data.risk_level;
    badge.style.color = color;
    recText.innerText = data.recommendation;

    // Render Risk Indicators
    factorsList.innerHTML = "";
    data.key_risk_factors.forEach(factor => {
        const li = document.createElement("li");
        li.innerHTML = `<i class="fa-solid ${data.risk_level === 'LOW RISK' ? 'fa-circle-check' : 'fa-triangle-exclamation'}" style="color:${color}"></i> ${factor}`;
        factorsList.appendChild(li);
    });
}

// Fetch Metrics & Initialize Charts
async function fetchMetricsAndCharts() {
    try {
        const res = await fetch("/metrics");
        if (!res.ok) return;
        
        const data = await res.json();
        renderBenchmarkTable(data);
        renderModelChart(data);
        renderShapChart(data);
    } catch (err) {
        console.warn("Could not fetch metrics for charts:", err);
    }
}

// Render Benchmark Comparison Table
function renderBenchmarkTable(data) {
    const tbody = document.querySelector("#benchmark-table tbody");
    tbody.innerHTML = "";

    Object.entries(data.models).forEach(([modelName, metrics]) => {
        const isBest = modelName === data.best_model_name;
        const tr = document.createElement("tr");
        if (isBest) tr.style.background = "rgba(59, 130, 246, 0.1)";

        tr.innerHTML = `
            <td style="font-weight:600;">
                ${modelName} ${isBest ? '<span style="color:#3b82f6; font-size:0.75rem;">(CHAMPION)</span>' : ''}
            </td>
            <td>${metrics.roc_auc}</td>
            <td>${metrics.pr_auc}</td>
            <td>${metrics.f1_score}</td>
            <td>${metrics.precision}</td>
            <td>${metrics.recall}</td>
        `;
        tbody.appendChild(tr);
    });
}

// Render ROC-AUC Comparison Bar Chart
function renderModelChart(data) {
    const ctx = document.getElementById("modelChart").getContext("2d");
    if (modelChartInstance) modelChartInstance.destroy();

    const labels = Object.keys(data.models);
    const scores = labels.map(k => data.models[k].roc_auc);

    modelChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "ROC-AUC Score",
                data: scores,
                backgroundColor: labels.map(l => l === data.best_model_name ? "rgba(59, 130, 246, 0.85)" : "rgba(99, 102, 241, 0.4)"),
                borderColor: labels.map(l => l === data.best_model_name ? "#3b82f6" : "#6366f1"),
                borderWidth: 1.5,
                borderRadius: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                y: { min: 0.5, max: 1.0, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
                x: { grid: { display: false }, ticks: { color: "#94a3b8" } }
            }
        }
    });
}

// Render SHAP Feature Importance Chart
function renderShapChart(data) {
    const ctx = document.getElementById("shapChart").getContext("2d");
    if (shapChartInstance) shapChartInstance.destroy();

    const shapEntries = Object.entries(data.feature_importances).reverse();
    const labels = shapEntries.map(e => e[0]);
    const values = shapEntries.map(e => e[1]);

    shapChartInstance = new Chart(ctx, {
        type: "bar",
        data: {
            labels: labels,
            datasets: [{
                label: "Mean |SHAP| Impact",
                data: values,
                backgroundColor: "rgba(16, 185, 129, 0.75)",
                borderColor: "#10b981",
                borderWidth: 1.5,
                borderRadius: 6
            }]
        },
        options: {
            indexAxis: "y",
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#94a3b8" } },
                y: { grid: { display: false }, ticks: { color: "#f8fafc" } }
            }
        }
    });
}

// Batch File Upload Handler
async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
        const res = await fetch("/predict-batch", {
            method: "POST",
            body: formData
        });

        if (!res.ok) {
            const err = await res.json();
            alert(`Batch Error: ${err.detail}`);
            return;
        }

        const data = await res.json();
        renderBatchResults(data);
    } catch (err) {
        alert(`Upload Failed: ${err.message}`);
    }
}

function renderBatchResults(data) {
    document.getElementById("batch-summary").style.display = "grid";
    document.getElementById("table-container").style.display = "block";

    document.getElementById("batch-low-count").innerText = data.low_risk_count;
    document.getElementById("batch-med-count").innerText = data.medium_risk_count;
    document.getElementById("batch-high-count").innerText = data.high_risk_count;

    const tbody = document.querySelector("#batch-results-table tbody");
    tbody.innerHTML = "";

    data.predictions.slice(0, 100).forEach(row => {
        const tr = document.createElement("tr");
        let badgeColor = "#10b981";
        if (row.risk_level === "MEDIUM RISK") badgeColor = "#f59e0b";
        if (row.risk_level === "HIGH RISK") badgeColor = "#f43f5e";

        tr.innerHTML = `
            <td>#${row.index + 1}</td>
            <td>${row.default_probability}</td>
            <td>${row.risk_score_percentage}%</td>
            <td style="color:${badgeColor}; font-weight:600;">${row.risk_level}</td>
            <td>${row.recommendation}</td>
        `;
        tbody.appendChild(tr);
    });
}
