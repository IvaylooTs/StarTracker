const ctx = document.getElementById('quaternionChart').getContext('2d');
const maxDataPoints = 100;

const chart = new Chart(ctx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { label: 'w', borderColor: 'red', data: [], fill: false },
      { label: 'x', borderColor: 'green', data: [], fill: false },
      { label: 'y', borderColor: 'blue', data: [], fill: false },
      { label: 'z', borderColor: 'yellow', data: [], fill: false },
    ]
  },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: { display: false },
      y: {
        suggestedMin: -1,
        suggestedMax: 1,
        title: { display: true, text: 'Quaternion Component Value' }
      }
    },
    animation: false
  }
});


function AddDataToChart(q) {
    const now = new Date().toLocaleTimeString();
    chart.data.labels.push(now);
    chart.data.datasets[0].data.push(q.w);
    chart.data.datasets[1].data.push(q.x);
    chart.data.datasets[2].data.push(q.y);
    chart.data.datasets[3].data.push(q.z);

    if (chart.data.labels.length > maxDataPoints) {
      chart.data.labels.shift();
      chart.data.datasets.forEach(dataset => dataset.data.shift());
    }
    chart.update();
}


