document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const newsInput = document.getElementById('news-input');
    const charCount = document.getElementById('char-count');
    const clearBtn = document.getElementById('clear-btn');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    const resultPlaceholder = document.getElementById('result-placeholder');
    const resultLoading = document.getElementById('result-loading');
    const resultDisplay = document.getElementById('result-display');
    
    const resultBadge = document.getElementById('result-badge');
    const resultVerdict = document.getElementById('result-verdict');
    const confidencePercentage = document.getElementById('confidence-percentage');
    const confidenceBar = document.getElementById('confidence-bar');
    
    const historyList = document.getElementById('history-list');
    const noHistoryMsg = document.getElementById('no-history-msg');

    let historyData = [];

    // Character Counter & Button Enable/Disable
    newsInput.addEventListener('input', () => {
        const textLength = newsInput.value.length;
        charCount.textContent = textLength.toLocaleString();
        
        if (textLength >= 10) {
            analyzeBtn.removeAttribute('disabled');
        } else {
            analyzeBtn.setAttribute('disabled', 'true');
        }
    });

    // Clear Button Action
    clearBtn.addEventListener('click', () => {
        newsInput.value = '';
        charCount.textContent = '0';
        analyzeBtn.setAttribute('disabled', 'true');
        
        // Reset Result cards to placeholder state
        resultDisplay.classList.add('hidden');
        resultLoading.classList.add('hidden');
        resultPlaceholder.classList.remove('hidden');
    });

    // Analyze Button Action
    analyzeBtn.addEventListener('click', () => {
        const text = newsInput.value.trim();
        if (text.length < 10) return;

        // Transition to Loading state
        resultPlaceholder.classList.add('hidden');
        resultDisplay.classList.add('hidden');
        resultLoading.classList.remove('hidden');

        // Simulate model inference time (1.2 seconds)
        setTimeout(() => {
            const analysis = performMockClassification(text);
            displayResult(analysis);
            addToHistory(text, analysis);
        }, 1200);
    });

    // Simple analysis heuristic for interactive demo
    function performMockClassification(text) {
        const lowerText = text.toLowerCase();
        
        // Words frequently associated with clickbait/fake news in classic datasets
        const clickbaitWords = [
            'shocking', 'unbelievable', 'conspiracy', 'secret exposed', 
            'miracle cure', 'aliens', 'won\'t believe', 'classified', 
            'illuminati', 'faked', 'scam', 'propaganda', 'hoax'
        ];
        
        // Words indicating standard factual reporting terms
        const factualWords = [
            'official', 'spokesman', 'reuters', 'announced', 'spokesperson',
            'study suggests', 'according to', 'researchers', 'confirmed by',
            'legislature', 'department of', 'ministry', 'signed a contract'
        ];

        let fakeScore = 0;
        let realScore = 0;

        clickbaitWords.forEach(word => {
            if (lowerText.includes(word)) fakeScore += 2;
        });

        factualWords.forEach(word => {
            if (lowerText.includes(word)) realScore += 1.5;
        });

        // Default confidence baseline
        let label = 'REAL';
        let confidence = 50 + Math.floor(Math.random() * 30); // 50% - 80%

        if (fakeScore > realScore) {
            label = 'FAKE';
            confidence = Math.min(65 + (fakeScore * 8) + Math.floor(Math.random() * 10), 99);
        } else if (realScore > fakeScore) {
            label = 'REAL';
            confidence = Math.min(60 + (realScore * 10) + Math.floor(Math.random() * 10), 98);
        } else {
            // Neutral/ambiguous - slight bias to random
            label = Math.random() > 0.55 ? 'REAL' : 'FAKE';
            confidence = 55 + Math.floor(Math.random() * 20);
        }

        return { label, confidence };
    }

    // Display Results in UI
    function displayResult(analysis) {
        resultLoading.classList.add('hidden');
        resultDisplay.classList.remove('hidden');

        // Reset badge classes
        resultBadge.className = 'badge';
        confidenceBar.className = 'meter-fill';

        if (analysis.label === 'REAL') {
            resultBadge.classList.add('real');
            resultBadge.textContent = 'REAL';
            resultVerdict.textContent = 'This article appears to be REAL';
            confidenceBar.classList.add('real-fill');
        } else {
            resultBadge.classList.add('fake');
            resultBadge.textContent = 'FAKE';
            resultVerdict.textContent = 'This article appears to be FAKE';
            confidenceBar.classList.add('fake-fill');
        }

        confidencePercentage.textContent = `${analysis.confidence}%`;
        confidenceBar.style.width = `${analysis.confidence}%`;
    }

    // Add to prediction history
    function addToHistory(text, analysis) {
        // Truncate snippet
        const snippet = text.length > 70 ? text.substring(0, 70) + '...' : text;
        
        // Add to history list array
        historyData.unshift({
            text: snippet,
            label: analysis.label,
            id: Date.now()
        });

        // Limit to last 5 entries
        if (historyData.length > 5) {
            historyData.pop();
        }

        renderHistory();
    }

    // Render history items
    function renderHistory() {
        if (historyData.length === 0) {
            noHistoryMsg.classList.remove('hidden');
            return;
        }

        noHistoryMsg.classList.add('hidden');
        
        // Clear current elements besides fallback message
        const currentItems = historyList.querySelectorAll('.history-item');
        currentItems.forEach(item => item.remove());

        // Create new history elements
        historyData.forEach(item => {
            const div = document.createElement('div');
            div.className = 'history-item';
            
            const badgeClass = item.label === 'REAL' ? 'real' : 'fake';
            const badgeText = item.label === 'REAL' ? '✅ Real' : '🚨 Fake';

            div.innerHTML = `
                <div class="history-text">${escapeHtml(item.text)}</div>
                <span class="history-badge ${badgeClass}">${badgeText}</span>
            `;
            
            historyList.appendChild(div);
        });
    }

    // Escape HTML to prevent XSS in mockup
    function escapeHtml(unsafe) {
        return unsafe
             .replace(/&/g, "&amp;")
             .replace(/</g, "&lt;")
             .replace(/>/g, "&gt;")
             .replace(/"/g, "&quot;")
             .replace(/'/g, "&#039;");
    }
});
