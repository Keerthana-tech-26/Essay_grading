document.addEventListener("DOMContentLoaded", () => {
  const { trendLabels, overallScores, avgScores, issueLabels, issueValues } = window.dashboardData;
  const ctx1 = document.getElementById('scoreTrend').getContext('2d');
  new Chart(ctx1, {
    type: 'line',
    data: {
      labels: trendLabels,
      datasets: [{
        label: 'Overall Score',
        data: overallScores,
        fill: false,
        borderColor: 'rgb(75, 192, 192)',
        tension: 0.25
      }]
    },
    options: {
      scales: {
        y: { min: 0, max: 100, title: { display: true, text: 'Score' } }
      }
    }
  });
  const subLabels = Object.keys(avgScores);
  const subValues = Object.values(avgScores);
  const ctx2 = document.getElementById('avgScores').getContext('2d');
  new Chart(ctx2, {
    type: 'bar',
    data: {
      labels: subLabels,
      datasets: [{
        label: 'Average',
        data: subValues,
        backgroundColor: 'rgba(54, 162, 235, 0.6)'
      }]
    },
    options: {
      indexAxis: 'y',
      scales: { x: { min: 0, max: 100 } }
    }
  });
  const ctx3 = document.getElementById('issuesChart').getContext('2d');
  new Chart(ctx3, {
    type: 'doughnut',
    data: {
      labels: issueLabels,
      datasets: [{
        data: issueValues,
        backgroundColor: [
          'rgba(255, 99, 132, 0.6)',
          'rgba(255, 159, 64, 0.6)',
          'rgba(255, 205, 86, 0.6)',
          'rgba(75, 192, 192, 0.6)',
          'rgba(54, 162, 235, 0.6)'
        ]
      }]
    },
    options: {
      plugins: {
        legend: { position: 'bottom' }
      }
    }
  });
});
