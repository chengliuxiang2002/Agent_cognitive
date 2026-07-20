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
      if (panelId === 'team') initTeamPanel();
      if (panelId === 'document') initDocumentPanel();
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

async function apiPost(path, data = {}) {
  const url = `/api/admin${path}`;
  try {
    const resp = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (e) {
    console.error(`API POST error: ${path}`, e);
    return null;
  }
}

// ─── 仪表盘 ──────────────────────────────────────────────

async function loadDashboard() {
  const [status, users, sim, online] = await Promise.all([
    apiGet('/status'),
    apiGet('/users'),
    apiGet('/simulator'),
    apiGet('/online-users'),
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

  if (sim) {
    const badge = document.getElementById('simStatus');
    if (badge) {
      badge.textContent = sim.running ? `模拟中 (${sim.sim_count}条)` : '已停止';
      badge.className = sim.running ? 'status-badge status-running' : 'status-badge status-warning';
    }
  }

  // 渲染在线用户
  if (online) {
    document.getElementById('onlineCount').textContent = online.online;
    const list = document.getElementById('onlineUsersList');
    if (online.users.length === 0) {
      list.innerHTML = '<span class="online-empty">暂无在线用户</span>';
    } else {
      list.innerHTML = online.users.map(u =>
        `<span class="online-user-tag">
          <span class="dot"></span>${u.name}(${u.user_id})
          <span class="time-ago">${u.seconds_ago}s前</span>
        </span>`
      ).join('');
    }
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

  console.log('Graph API response:', data);

  if (!data || data.error) {
    console.error('Graph data error:', data?.error);
    document.getElementById('graphContainer').innerHTML =
      `<div class="empty-state">数据加载失败: ${data?.error || '未知错误'}</div>`;
    return;
  }

  if (!data.nodes || data.nodes.length === 0) {
    document.getElementById('graphContainer').innerHTML =
      '<div class="empty-state">暂无数据，请先填充种子数据或输入用户ID查询</div>';
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
    driving_mode: '#27AE60',
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

// ─── FE-1: 团队管理 ──────────────────────────────────────

let _currentTeamId = null;

function initTeamPanel() {
  document.getElementById('teamLoadBtn').addEventListener('click', loadTeams);
  // 默认加载
  setTimeout(() => loadTeams(), 500);
}

async function loadTeams() {
  const userId = document.getElementById('teamUserId').value.trim();
  if (!userId) { alert('请输入用户ID'); return; }

  const data = await apiGet(`/teams/${userId}`);
  if (!data || data.error) {
    document.getElementById('teamList').innerHTML = `<div class="empty-state">加载失败: ${data?.error || '未知错误'}</div>`;
    return;
  }

  const teams = data.teams || [];
  const list = document.getElementById('teamList');

  if (teams.length === 0) {
    list.innerHTML = '<div class="empty-state">该用户暂无所属团队，点击"创建团队"新建</div>';
    return;
  }

  list.innerHTML = teams.map((team, idx) => `
    <div class="team-card" onclick="showTeamDetail('${team.id}')">
      <div class="team-card-header">
        <span class="team-card-name">${team.name}</span>
        <span class="team-card-dept">${team.department || '--'}</span>
      </div>
      <div class="team-card-meta">
        <span>👥 ${(team.members || []).length} 人</span>
        <span>📝 ${team.memory_count || 0} 条记忆</span>
        <span>🕐 ${team.created_at ? new Date(team.created_at).toLocaleDateString() : '--'}</span>
      </div>
      <div class="team-card-desc">${team.description || '暂无描述'}</div>
    </div>
  `).join('');
}

function showCreateTeam() {
  document.getElementById('createTeamForm').style.display = 'block';
  document.getElementById('teamDetail').style.display = 'none';
}

function hideCreateTeam() {
  document.getElementById('createTeamForm').style.display = 'none';
}

async function doCreateTeam() {
  const name = document.getElementById('newTeamName').value.trim();
  if (!name) { alert('请输入团队名称'); return; }

  const membersText = document.getElementById('newTeamMembers').value.trim();
  const members = [];
  if (membersText) {
    membersText.split('\n').forEach(line => {
      const parts = line.trim().split(',');
      if (parts.length >= 1) {
        members.push({
          user_id: parts[0].trim(),
          role: parts[1]?.trim() || 'member',
          permission: parts[2]?.trim() || 'view',
        });
      }
    });
  }

  const data = await apiPost('/teams', {
    name: name,
    description: document.getElementById('newTeamDesc').value.trim(),
    department: document.getElementById('newTeamDept').value.trim(),
    members: members,
    created_by: document.getElementById('teamUserId').value.trim(),
  });

  if (data && data.success) {
    alert('团队创建成功！');
    hideCreateTeam();
    // 重置表单
    document.getElementById('newTeamName').value = '';
    document.getElementById('newTeamDesc').value = '';
    document.getElementById('newTeamDept').value = '';
    document.getElementById('newTeamMembers').value = '';
    // 刷新列表
    await loadTeams();
  } else {
    alert('创建失败: ' + (data?.error || '未知错误'));
  }
}

async function showTeamDetail(teamId) {
  _currentTeamId = teamId;
  document.getElementById('teamDetail').style.display = 'block';
  document.getElementById('createTeamForm').style.display = 'none';

  // 获取团队详情
  const userId = document.getElementById('teamUserId').value.trim();
  const data = await apiGet(`/teams/${userId}`);
  if (!data || !data.teams) return;

  const team = data.teams.find(t => t.id === teamId);
  if (!team) return;

  document.getElementById('teamDetailName').textContent = team.name;
  document.getElementById('teamDetailMeta').textContent = `${team.department || '--'} | 创建者: ${team.created_by || '--'}`;

  const members = team.members || [];
  document.getElementById('teamDetailMembers').innerHTML = `
    <h4>团队成员 (${members.length}人)</h4>
    <div class="member-list">
      ${members.map(m => `
        <div class="member-tag">
          <span class="member-id">${m.user_id}</span>
          <span class="member-role">${m.role || 'member'}</span>
          <span class="member-perm ${m.permission}">${m.permission}</span>
        </div>
      `).join('')}
    </div>
  `;

  // 加载团队记忆
  await loadTeamMemories(teamId, userId);
}

async function loadTeamMemories(teamId, userId) {
  const data = await apiGet(`/teams/${teamId}/memories`, { user_id: userId });
  const list = document.getElementById('teamMemoriesList');

  if (!data || data.error) {
    list.innerHTML = `<p class="empty-hint">加载失败: ${data?.error || ''}</p>`;
    return;
  }

  const items = data.items || [];
  if (items.length === 0) {
    list.innerHTML = '<p class="empty-hint">暂无团队记忆</p>';
    return;
  }

  list.innerHTML = items.map(item => `
    <div class="memory-item">
      <div class="memory-item-header">
        <span class="memory-item-title">${item.title || '无标题'}</span>
        <span class="memory-item-type">${item.memory_type || 'general'}</span>
      </div>
      <div class="memory-item-content">${JSON.stringify(item.content || {}, null, 2)}</div>
      <div class="memory-item-meta">
        <span>${item.created_by || '--'}</span>
        <span>${item.created_at ? new Date(item.created_at).toLocaleString() : '--'}</span>
        ${item.tags ? item.tags.map(t => `<span class="tag">${t}</span>`).join('') : ''}
      </div>
    </div>
  `).join('');
}

function showCreateTeamMemory() {
  document.getElementById('createTeamMemoryForm').style.display = 'block';
}

function hideCreateTeamMemory() {
  document.getElementById('createTeamMemoryForm').style.display = 'none';
}

async function doCreateTeamMemory() {
  if (!_currentTeamId) { alert('请先选择团队'); return; }

  const title = document.getElementById('newMemTitle').value.trim();
  if (!title) { alert('请输入记忆标题'); return; }

  let content = {};
  try {
    const contentStr = document.getElementById('newMemContent').value.trim();
    if (contentStr) content = JSON.parse(contentStr);
  } catch (e) {
    alert('内容JSON格式错误');
    return;
  }

  const data = await apiPost(`/teams/${_currentTeamId}/memories`, {
    title: title,
    memory_type: document.getElementById('newMemType').value,
    content: content,
    created_by: document.getElementById('teamUserId').value.trim(),
  });

  if (data && data.success) {
    alert('记忆创建成功！');
    hideCreateTeamMemory();
    document.getElementById('newMemTitle').value = '';
    document.getElementById('newMemContent').value = '';
    await loadTeamMemories(_currentTeamId, document.getElementById('teamUserId').value.trim());
  } else {
    alert('创建失败: ' + (data?.error || '未知错误'));
  }
}

// ─── FE-4: 文档上下文 ────────────────────────────────────

function initDocumentPanel() {
  document.getElementById('docLoadBtn').addEventListener('click', loadDocuments);
  // 默认加载
  setTimeout(() => loadDocuments(), 500);
}

async function loadDocuments() {
  const userId = document.getElementById('docUserId').value.trim();
  if (!userId) { alert('请输入用户ID'); return; }

  const data = await apiGet(`/documents/${userId}/recent`);
  if (!data || data.error) {
    document.getElementById('docList').innerHTML = `<div class="empty-state">加载失败: ${data?.error || ''}</div>`;
    return;
  }

  const docs = data.documents || [];
  const list = document.getElementById('docList');

  if (docs.length === 0) {
    list.innerHTML = '<div class="empty-state">该用户暂无文档记录</div>';
    return;
  }

  list.innerHTML = docs.map(doc => `
    <div class="doc-item" onclick="showDocumentDetail('${doc.id}')">
      <div class="doc-item-icon ${doc.file_format}">${getDocIcon(doc.file_format)}</div>
      <div class="doc-item-info">
        <div class="doc-item-name">${doc.file_name}</div>
        <div class="doc-item-summary">${doc.content_summary || '--'}</div>
      </div>
      <div class="doc-item-meta">
        <span>✏️ ${doc.edit_count || 0}次</span>
        <span>🕐 ${doc.last_accessed_at ? formatTimeAgo(doc.last_accessed_at) : '--'}</span>
      </div>
    </div>
  `).join('');

  // 存储文档数据供详情使用
  window._documents = docs;
}

function getDocIcon(format) {
  const icons = { docx: '📄', doc: '📄', pdf: '📕', txt: '📝', xlsx: '📊' };
  return icons[format] || '📄';
}

function formatTimeAgo(isoStr) {
  const diff = Date.now() - new Date(isoStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 60) return `${mins}分钟前`;
  const hours = Math.floor(mins / 60);
  if (hours < 24) return `${hours}小时前`;
  return `${Math.floor(hours / 24)}天前`;
}

function showDocumentDetail(docId) {
  const docs = window._documents || [];
  const doc = docs.find(d => d.id === docId);
  if (!doc) return;

  document.getElementById('docDetail').style.display = 'block';
  document.getElementById('docDetailName').textContent = doc.file_name;
  document.getElementById('docDetailFormat').textContent = doc.file_format?.toUpperCase() || '--';
  document.getElementById('docDetailSummary').textContent = doc.content_summary || '暂无摘要';
  document.getElementById('docDetailEdits').textContent = `${doc.edit_count || 0} 次`;
  document.getElementById('docDetailTime').textContent = formatEditTime(doc.total_edit_time || 0);
  document.getElementById('docDetailLastAccess').textContent = doc.last_accessed_at ? new Date(doc.last_accessed_at).toLocaleString() : '--';
  document.getElementById('docDetailSessions').textContent = (doc.associated_sessions || []).length + ' 个';

  const keywords = doc.keywords || [];
  document.getElementById('docDetailKeywords').innerHTML = keywords.length > 0
    ? keywords.map(k => `<span class="tag">${k}</span>`).join('')
    : '<span class="empty-hint">暂无关键词</span>';
}

function formatEditTime(seconds) {
  if (seconds < 60) return `${seconds}秒`;
  const mins = Math.floor(seconds / 60);
  if (mins < 60) return `${mins}分钟`;
  const hours = Math.floor(mins / 60);
  return `${hours}小时${mins % 60}分钟`;
}

// ─── FE-7: 数据导出 ──────────────────────────────────────

function showExportModal() {
  const userId = document.getElementById('radarUserId').value.trim();
  if (userId) {
    document.getElementById('exportUserId').value = userId;
  }
  document.getElementById('exportModal').style.display = 'flex';
  document.getElementById('exportResult').style.display = 'none';
}

function closeExportModal() {
  document.getElementById('exportModal').style.display = 'none';
}

async function doExport() {
  const userId = document.getElementById('exportUserId').value.trim();
  if (!userId) { alert('请输入用户ID'); return; }

  const format = document.getElementById('exportFormat').value;
  const categories = [];
  document.querySelectorAll('#exportResult').closest('.modal-body').querySelectorAll('.checkbox-group input:checked').forEach(cb => {
    // 从模态框内获取选中的checkbox
  });
  // 重新获取
  document.querySelectorAll('#exportModal .checkbox-group input:checked').forEach(cb => {
    categories.push(cb.value);
  });

  const resultDiv = document.getElementById('exportResult');
  resultDiv.style.display = 'block';
  resultDiv.innerHTML = '<div class="loading">正在导出数据...</div>';

  const data = await apiPost('/export', {
    user_id: userId,
    format: format,
    categories: categories.length > 0 ? categories : undefined,
  });

  if (!data || !data.success) {
    resultDiv.innerHTML = `<div class="error">导出失败: ${data?.error || '未知错误'}</div>`;
    return;
  }

  if (data.data?.async) {
    resultDiv.innerHTML = `<div class="success">异步导出已启动<br>任务ID: ${data.data.task_id}<br>预计大小: ${data.data.estimated_size_mb}MB<br>${data.data.message}</div>`;
    return;
  }

  if (format === 'csv') {
    // 下载CSV文件
    const blob = new Blob([data.data.content], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `memory_export_${userId}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    resultDiv.innerHTML = `<div class="success">CSV导出完成，共 ${data.data.total_items} 条记录</div>`;
  } else {
    // 下载JSON文件
    const blob = new Blob([JSON.stringify(data.data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `memory_export_${userId}.json`;
    a.click();
    URL.revokeObjectURL(url);
    resultDiv.innerHTML = `<div class="success">JSON导出完成，共 ${data.data.total_items} 条记录</div>`;
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

  // 初始化新面板 (FE-1, FE-4)
  initTeamPanel();
  initDocumentPanel();

  // 导出弹窗点击外部关闭
  document.getElementById('exportModal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeExportModal();
  });

  // 启动自动刷新
  startAutoRefresh();
});