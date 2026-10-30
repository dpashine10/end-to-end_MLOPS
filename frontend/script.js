const config = window.SMS_SPAM_CONFIG || {};
const apiBaseUrl = (config.apiBaseUrl || '').replace(/\/$/, '');
const form = document.querySelector('#checker-form');
const message = document.querySelector('#message');
const counter = document.querySelector('#counter');
const result = document.querySelector('#result');
const submitButton = document.querySelector('#submit-button');
const apiStatus = document.querySelector('#api-status');

function updateCounter() {
  counter.textContent = `${message.value.length} / ${message.maxLength}`;
}

function showResult(title, detail, score, kind) {
  result.replaceChildren();
  result.hidden = false;
  result.className = `result ${kind}`;
  const heading = document.createElement('h3');
  heading.className = 'result-title';
  heading.textContent = title;
  const description = document.createElement('p');
  description.className = 'result-detail';
  description.textContent = detail;
  result.append(heading, description);
  if (score) {
    const scoreElement = document.createElement('p');
    scoreElement.className = 'score';
    scoreElement.textContent = `Spam probability: ${score}`;
    result.append(scoreElement);
  }
}

async function updateApiStatus() {
  try {
    const response = await fetch(`${apiBaseUrl}/health`);
    const health = await response.json();
    const loaded = response.ok && health.model_loaded;
    apiStatus.textContent = loaded ? 'Model ready' : 'Model unavailable';
    apiStatus.classList.toggle('ready', loaded);
  } catch {
    apiStatus.textContent = 'API unavailable';
    apiStatus.classList.remove('ready');
  }
}

message.addEventListener('input', updateCounter);
document.querySelector('#clear-button').addEventListener('click', () => {
  message.value = '';
  result.hidden = true;
  updateCounter();
  message.focus();
});
document.querySelectorAll('.example').forEach((button) => button.addEventListener('click', () => {
  message.value = button.dataset.message;
  updateCounter();
  message.focus();
}));

form.addEventListener('submit', async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  submitButton.textContent = 'Checking…';
  result.hidden = true;
  try {
    const response = await fetch(`${apiBaseUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: message.value }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || 'The API could not process this message.');
    const spam = payload.predicted_class === 'spam';
    showResult(
      spam ? 'Likely spam' : 'Likely not spam',
      `Model version ${payload.model_version}. This score is an estimate, not a guarantee.`,
      `${(payload.spam_probability * 100).toFixed(1)}%`,
      spam ? 'spam' : 'not-spam',
    );
  } catch (error) {
    showResult('Unable to check message', error.message, '', 'error');
  } finally {
    submitButton.disabled = false;
    submitButton.innerHTML = 'Check Message <span aria-hidden="true">&#8594;</span>';
  }
});

updateCounter();
updateApiStatus();
