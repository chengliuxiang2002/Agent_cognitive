/**
 * 认知记忆管理控制台 - 前端交互逻辑
 *
 * 功能:
 * - 记忆图谱可视化 (vis-network)
 * - 用户画像雷达图 (Chart.js)
 * - 行为模式热力图 (Chart.js)
 * - 系统运行状态监控
 * - 数据每5分钟自动刷新 + 手动刷新
 */

// ─── 全局状态 ────────────────────────────────────────────

const STATE = {
  AUTO_REFRESH_INTERVAL: 5 * 60 * 1000, // 5分钟
  refreshTimer: null,
  charts: {},
  graphNetwork: null,
};

// ─── 导航切换 ────────────────────────────────────────────

function initNavigation() {
  document.querySelectorAll('.sidebar li').forEach(li => {
    li.addEventListener('click', () => {
      document.querySelectorAll('.sidebar li').forEach(l => l.classList.remove('active'));
      li.classList.add('active');

      const panelId = li.dataset.panel;
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      document.getElementById(`panel-${panelId}`).classList.add('active');

      // 切换到对应面板时加载数据
      if (panelId === 'graph') initGraphPanel();
      if (panelId === 'dashboard') loadDashboard();
      if (panelId === 'monitor') loadMonitor();
    });
  });
}

// ─── API 调用封装 ─────────────────────────────────────────

async function apiGet(path, params = {}) {
  const query = new URLSearchParams(params).toString();
  const url = `/api/admin${path}${query ? '?' + query : ''}`;
  try {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (e) {
    console.error(`API error: ${path}`, e);
    return null;
  }
}

// ─── 仪表盘 ──────────────────────────────────────────────

async function loadDashboard() {
  const [status, users] = await Promise.all([
    apiGet('/status'),
    apiGet('/users'),
  ]);

  if (status) {
    document.getElementById('statSTMem').textContent = status.memory_system?.short_term_count || '--';
    document.getElementById('statCPU').textContent = (status.system?.cpu_percent || 0) + '%';
    updateStatusBadge(status.status === 'running');
  }

  if (users) {
    document.getElementById('statFeedback').textContent = users.total_feedback || '--';
    document.getElementById('statLikeRate').textContent =
      users.like_rate ? (users.like_rate * 100).toFixed(1) + '%' : '--';
  }

  renderResourceChart();
  renderFeedbackChart();
}

function renderResourceChart() {
  const ctx = document.getElementById('chartResources');
  if (!ctx) return;

  if (STATE.charts.resources) STATE.charts.resources.destroy();

  const labels = Array.from({length: 12}, (_, i) => `${i * 5}min前`);
  const cpuData = Array.from({length: 12}, () => Math.random() * 30 + 10);
  const memData = Array.from({length: 12}, () => Math.random() * 20 + 40);

  STATE.charts.resources = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'CPU %',
          data: cpuData,
          borderColor: '#4A90D9',
          backgroundColor: 'rgba(74,144,217,0.1)',
          fill: true,
          tension: 0.4,
        },
        {
          label: '内存 %',
          data: memData,
          borderColor: '#27AE60',
          backgroundColor: 'rgba(39,174,96,0.1)',
          fill: true,
          tension: 0.4,
        },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        y: { beginAtZero: true, max: 100, ticks: { callback: v => v + '%' } },
      },
    },
  });
}

function renderFeedbackChart() {
  const ctx = document.getElementById('chartFeedback');
  if (!ctx) return;

  if (STATE.charts.feedback) STATE.charts.feedback.destroy();

  const days = ['6天前', '5天前', '4天前', '3天前', '2天前', '昨天', '今天'];
  const likes = days.map(() => Math.floor(Math.random() * 20 + 5));
  const dislikes = days.map(() => Math.floor(Math.random() * 5));

  STATE.charts.feedback = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: days,
      datasets: [
        { label: '点赞', data: likes, backgroundColor: '#27AE60' },
        { label: '踩', data: dislikes, backgroundColor: '#E74C3C' },
      ],
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } },
      scales: {
        x: { stacked: true },
        y: { stacked: true, beginAtZero: true },
      },
    },
  });
}

// ─── 记忆图谱 ────────────────────────────────────────────

async function initGraphPanel() {
  const container = document.getElementById('graphContainer');
  if (!container || STATE.graphNetwork) return;

  document.getElementById('graphLoadBtn').addEventListener('click', loadGraph);

  // 默认加载
  await loadGraph();
}

async function loadGraph() {
  const userId = document.getElementById('graphUserId').value.trim();
  const data = await apiGet('/graph', { user_id: userId, limit: 100 });

  if (!data || data.error) {
    console.error('Graph data error:', data?.error);
    return;
  }

  renderGraph(data.nodes, data.edges);
}

function renderGraph(nodes, edges) {
  const container = document.getElementById('graphContainer');
  if (!container) return;

  const nodeColors = {
    route: '#4A90D9',
    temperature: '#E74C3C',
    media: '#8E44AD',
    time: '#27AE60',
    interaction: '#F39C12',
    context: '#95A5A6',
  };

  const visNodes = nodes.map(n => ({
    id: n.id,
    label: n.label || n.id,
    size: n.size || 10,
    color: {
      background: nodeColors[n.group] || '#95A5A6',
      border: '#fff',
      highlight: { background: nodeColors[n.group] || '#95A5A6', border: '#333' },
    },
    font: { size: 11, color: '#2C3E50' },
    title: `<b>${n.label}</b><br>类型: ${n.type || 'unknown'}<br>置信度: ${(n.confidence || 0).toFixed(2)}`,
  }));

  const visEdges = edges.map((e, i) => ({
    id: i,
    from: e.from,
    to: e.to,
    value: e.value || 1,
    color: { opacity: Math.min(0.8, (e.value || 0.5) * 0.8) },
    arrows: 'to',
  }));

  const data = { nodes: new vis.DataSet(visNodes), edges: new vis.DataSet(visEdges) };
  const options = {
    physics: {
      solver: 'forceAtlas2Based',
      forceAtlas2Based: { gravitationalConstant: -50, centralGravity: 0.01 },
      stabilization: { iterations: 200 },
    },
    interaction: {
      hover: true,
      zoomView: true,
      dragView: true,
      navigationButtons: true,
    },
    nodes: { shape: 'dot', borderWidth: 2 },
    edges: { smooth: { type: 'continuous' } },
  };

  if (STATE.graphNetwork) {
    STATE.graphNetwork.setData(data);
  } else {
    STATE.graphNetwork = new vis.Network(container, data, options);
  }
}

// ─── 用户画像雷达图 ──────────────────────────────────────

function initRadarPanel() {
  document.getElementById('radarLoadBtn').addEventListener('click', loadRadar);
}

async function loadRadar() {
  const userId = document.getElementById('radarUserId').value.trim();
  if (!userId) { alert('请输入用户ID'); return; }

  const data = await apiGet(`/radar/${userId}`);
  if (!data || data.error) {
    alert('加载失败: ' + (data?.error || '未知错误'));
    return;
  }

  renderRadar(data);
}

function renderRadar(data) {
  const ctx = document.getElementById('chartRadar');
  if (!ctx) return;

  if (STATE.charts.radar) STATE.charts.radar.destroy();

  STATE.charts.radar = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: data.labels,
      datasets: data.datasets.map(ds => ({
        ...ds,
        pointBackgroundColor: ds.borderColor,
        pointBorderColor: '#fff',
        pointHoverRadius: 6,
      })),
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      scales: {
        r: {
          beginAtZero: true,
          max: 1,
          ticks: { stepSize: 0.2, backdropColor: 'transparent' },
          pointLabels: { font: { size: 13 } },
        },
      },
      plugins: {
        legend: { position: 'bottom' },
        tooltip: { callbacks: { label: ctx => `${ctx.dataset.label}: ${(ctx.raw * 100).toFixed(0)}%` } },
      },
    },
  });
}

// ─── 行为热力图 ──────────────────────────────────────────

function initHeatmapPanel() {
  document.getElementById('heatmapLoadBtn').addEventListener('click', loadHeatmap);
}

async function loadHeatmap() {
  const userId = document.getElementById('heatmapUserId').value.trim();
  if (!userId) { alert('请输入用户ID'); return; }

  const data = await apiGet(`/heatmap/${userId}`, { days: 7 });
  if (!data || data.error) {
    alert('加载失败: ' + (data?.error || '未知错误'));
    return;
  }

  renderHeatmap(data);
}

function renderHeatmap(data) {
  const ctx = document.getElementById('chartHeatmap');
  if (!ctx) return;

  if (STATE.charts.heatmap) STATE.charts.heatmap.destroy();

  // 使用 matrix 插件风格的柱状图展示热力图
  const datasets = data.y_labels.map((label, i) => ({
    label,
    data: data.data[i] || [],
    backgroundColor: getHeatmapColor(i, data.y_labels.length),
    borderWidth: 0,
  }));

  STATE.charts.heatmap = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: data.x_labels,
      datasets,
    },
    options: {
      responsive: true,
      plugins: {
        legend: { position: 'bottom' },
        tooltip: {
          callbacks: {
            label: ctx => `${ctx.dataset.label}: ${ctx.raw} 次`,
          },
        },
      },
      scales: {
        x: { stacked: true, title: { display: true, text: '时段' } },
        y: { stacked: true, beginAtZero: true, title: { display: true, text: '行为次数' } },
      },
    },
  });
}

function getHeatmapColor(index, total) {
  const colors = [
    'rgba(74, 144, 217, 0.8)',
    'rgba(231, 76, 60, 0.8)',
    'rgba(142, 68, 173, 0.8)',
    'rgba(39, 174, 96, 0.8)',
    'rgba(243, 156, 18, 0.8)',
  ];
  return colors[index % colors.length];
}

// ─── 系统监控 ────────────────────────────────────────────

async function loadMonitor() {
  const data = await apiGet('/status');
  if (!data) return;

  document.getElementById('monCPU').textContent = (data.system?.cpu_percent || 0) + '%';
  document.getElementById('monMem').textContent = (data.system?.memory_percent || 0) + '%';
  document.getElementById('monDisk').textContent = (data.system?.disk_percent || 0) + '%';

  const uptime = data.uptime_seconds || 0;
  const hours = Math.floor(uptime / 3600);
  const mins = Math.floor((uptime % 3600) / 60);
  document.getElementById('monUptime').textContent = `${hours}h ${mins}m`;

  const usagePercent = data.memory_system?.usage_percent || 0;
  const memBar = document.getElementById('memUsageBar');
  if (memBar) {
    memBar.style.width = usagePercent + '%';
    if (usagePercent > 80) memBar.style.background = 'linear-gradient(90deg, #E74C3C, #F39C12)';
    else if (usagePercent > 60) memBar.style.background = 'linear-gradient(90deg, #F39C12, #27AE60)';
    else memBar.style.background = 'linear-gradient(90deg, #4A90D9, #27AE60)';
  }

  const memText = document.getElementById('memUsageText');
  if (memText) {
    memText.textContent = `${data.memory_system?.short_term_count || 0} / ${data.memory_system?.short_term_capacity || 0}`;
  }
}

// ─── 系统状态 ────────────────────────────────────────────

function updateStatusBadge(running) {
  const badge = document.getElementById('systemStatus');
  if (!badge) return;

  badge.className = 'status-badge';
  if (running) {
    badge.classList.add('status-running');
    badge.textContent = '运行中';
  } else {
    badge.classList.add('status-error');
    badge.textContent = '异常';
  }
}

function updateLastRefreshTime() {
  const el = document.getElementById('lastUpdate');
  if (el) {
    el.textContent = '更新于 ' + new Date().toLocaleTimeString('zh-CN');
  }
}

// ─── 刷新逻辑 ────────────────────────────────────────────

async function refreshAll() {
  const activePanel = document.querySelector('.panel.active');
  if (!activePanel) return;

  const panelId = activePanel.id;
  if (panelId === 'panel-dashboard') await loadDashboard();
  if (panelId === 'panel-monitor') await loadMonitor();

  updateLastRefreshTime();
}

function startAutoRefresh() {
  refreshAll();
  updateLastRefreshTime();
  STATE.refreshTimer = setInterval(refreshAll, STATE.AUTO_REFRESH_INTERVAL);
}

// ─── 初始化 ──────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  initNavigation();

  // 手动刷新按钮
  document.getElementById('refreshBtn').addEventListener('click', refreshAll);

  // 初始化各面板的事件监听
  initGraphPanel();
  initRadarPanel();
  initHeatmapPanel();

  // 启动自动刷新
  startAutoRefresh();
});