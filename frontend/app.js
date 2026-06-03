// RAGita Frontend JavaScript

const API_BASE_URL = 'http://localhost:8000';

let currentPersona = 'versatile';
let personas = [];

// Initialize app
document.addEventListener('DOMContentLoaded', async () => {
    await loadPersonas();
    setupEventListeners();
    await checkAPIHealth();
});

// Load available personas
async function loadPersonas() {
    try {
        const response = await fetch(`${API_BASE_URL}/personas`);
        const data = await response.json();
        personas = data.personas;
        renderPersonaButtons();
    } catch (error) {
        console.error('Error loading personas:', error);
        showError('Failed to load personas. Using default options.');
    }
}

// Render persona buttons
function renderPersonaButtons() {
    const container = document.getElementById('personaButtons');
    container.innerHTML = personas.map(persona => `
        <button 
            class="persona-btn px-4 py-2 rounded-lg transition-all ${persona.id === currentPersona ? 'bg-purple-600 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'}"
            data-persona="${persona.id}"
            title="${persona.description}"
        >
            ${persona.name}
        </button>
    `).join('');

    // Add click handlers
    document.querySelectorAll('.persona-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            currentPersona = e.target.dataset.persona;
            renderPersonaButtons(); // Re-render to update active state
        });
    });
}

// Setup event listeners
function setupEventListeners() {
    // Submit button
    document.getElementById('submitBtn').addEventListener('click', handleSubmit);

    // Enter key in textarea
    document.getElementById('queryInput').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            handleSubmit();
        }
    });

    // Example queries
    document.querySelectorAll('.example-query').forEach(btn => {
        btn.addEventListener('click', (e) => {
            document.getElementById('queryInput').value = e.target.textContent;
        });
    });

    // Stats modal
    document.getElementById('statsBtn').addEventListener('click', openStatsModal);
    document.getElementById('closeStatsBtn').addEventListener('click', closeStatsModal);

    // About modal
    document.getElementById('aboutBtn').addEventListener('click', openAboutModal);
    document.getElementById('closeAboutBtn').addEventListener('click', closeAboutModal);

    // Close modals on outside click
    document.getElementById('statsModal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeStatsModal();
    });
    document.getElementById('aboutModal').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) closeAboutModal();
    });
}

// Check API health
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        const data = await response.json();
        console.log('API Status:', data.status);
    } catch (error) {
        console.error('API health check failed:', error);
        showError('Cannot connect to RAGita API. Please ensure the backend is running on port 8000.');
    }
}

// Handle query submission
async function handleSubmit() {
    const query = document.getElementById('queryInput').value.trim();
    
    if (!query) {
        showError('Please enter a question.');
        return;
    }

    // Show loading state
    setLoading(true);
    hideError();
    hideResponse();

    try {
        const response = await fetch(`${API_BASE_URL}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                query: query,
                persona: currentPersona,
                top_k: 15,
                rerank_k: 8
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to process query');
        }

        const data = await response.json();
        displayResponse(data);
        
        // Hide welcome card after first query
        document.getElementById('welcomeCard').classList.add('hidden');
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'An error occurred while processing your query. Please try again.');
    } finally {
        setLoading(false);
    }
}

// Display response
function displayResponse(data) {
    const responseArea = document.getElementById('responseArea');
    const answerText = document.getElementById('answerText');
    const responseTime = document.getElementById('responseTime');
    const confidenceBadge = document.getElementById('confidenceBadge');
    const verificationStats = document.getElementById('verificationStats');
    const citationsGrid = document.getElementById('citationsGrid');

    // Answer text
    answerText.innerHTML = formatAnswer(data.answer);

    // Response time
    responseTime.textContent = `${data.response_time.toFixed(2)}s`;

    // Confidence badge
    const confidence = data.confidence_score;
    confidenceBadge.textContent = `${(confidence * 100).toFixed(0)}% Confidence`;
    confidenceBadge.className = `px-3 py-1 rounded-full text-xs font-semibold ${
        confidence >= 0.9 ? 'bg-green-100 text-green-800' :
        confidence >= 0.7 ? 'bg-yellow-100 text-yellow-800' :
        'bg-red-100 text-red-800'
    }`;

    // Verification stats
    const vs = data.verification_summary;
    verificationStats.textContent = `${vs.supported_claims}/${vs.total_claims} claims verified`;

    // Citations
    citationsGrid.innerHTML = data.citations.map((citation, index) => `
        <div class="citation-card p-4 bg-white rounded-lg border border-gray-200 shadow-sm">
            <div class="flex items-center justify-between mb-2">
                <span class="text-sm font-bold text-purple-600">${citation.verse}</span>
                <span class="text-xs text-gray-500">#${index + 1}</span>
            </div>
            <p class="text-sm text-gray-700 gita-font leading-relaxed">
                ${citation.passage_preview}
            </p>
        </div>
    `).join('');

    // Show response area with animation
    responseArea.classList.remove('hidden');
    responseArea.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Format answer text
function formatAnswer(text) {
    // Replace newlines with paragraphs
    const paragraphs = text.split('\n\n').filter(p => p.trim());
    return paragraphs.map(p => `<p class="mb-4">${escapeHtml(p)}</p>`).join('');
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// Show/hide loading state
function setLoading(loading) {
    const submitBtn = document.getElementById('submitBtn');
    const submitText = document.getElementById('submitText');
    const loadingIndicator = document.getElementById('loadingIndicator');

    if (loading) {
        submitBtn.disabled = true;
        submitText.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');
    } else {
        submitBtn.disabled = false;
        submitText.classList.remove('hidden');
        loadingIndicator.classList.add('hidden');
    }
}

// Show error
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    const errorText = document.getElementById('errorText');
    
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
    errorMessage.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Hide error
function hideError() {
    document.getElementById('errorMessage').classList.add('hidden');
}

// Hide response
function hideResponse() {
    document.getElementById('responseArea').classList.add('hidden');
}

// Stats modal
async function openStatsModal() {
    const modal = document.getElementById('statsModal');
    const content = document.getElementById('statsContent');

    modal.classList.remove('hidden');

    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const data = await response.json();

        content.innerHTML = `
            <div class="space-y-3">
                <div class="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                    <span class="text-gray-700 font-medium">Total Verses</span>
                    <span class="text-purple-600 font-bold">${data.total_verses}</span>
                </div>
                <div class="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                    <span class="text-gray-700 font-medium">Indexed Verses</span>
                    <span class="text-purple-600 font-bold">${data.indexed_verses}</span>
                </div>
                <div class="flex justify-between items-center p-3 bg-purple-50 rounded-lg">
                    <span class="text-gray-700 font-medium">Available Personas</span>
                    <span class="text-purple-600 font-bold">${data.available_personas.length}</span>
                </div>
                <div class="flex justify-between items-center p-3 bg-green-50 rounded-lg">
                    <span class="text-gray-700 font-medium">System Status</span>
                    <span class="text-green-600 font-bold uppercase">${data.system_status}</span>
                </div>
            </div>
        `;
    } catch (error) {
        content.innerHTML = `<p class="text-red-600">Failed to load statistics.</p>`;
    }
}

function closeStatsModal() {
    document.getElementById('statsModal').classList.add('hidden');
}

// About modal
function openAboutModal() {
    document.getElementById('aboutModal').classList.remove('hidden');
}

function closeAboutModal() {
    document.getElementById('aboutModal').classList.add('hidden');
}
