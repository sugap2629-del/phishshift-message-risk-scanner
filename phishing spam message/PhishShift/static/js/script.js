/**
 * PhishShift - Client-Side Threat Inspector Script
 * Handles real-time lexical counting, dynamic scan submission animation,
 * interactive mobile drawer, and clipboard integration.
 */

document.addEventListener('DOMContentLoaded', () => {
    initMessageCounter();
    initScanFormSubmission();
    initMobileNav();
    initKeybindings();
});

/**
 * Real-time character and word counters for the input textarea.
 */
function initMessageCounter() {
    const textarea = document.getElementById('message_text');
    const charCount = document.getElementById('char-count');
    const wordCount = document.getElementById('word-count');

    if (!textarea || !charCount || !wordCount) return;

    function updateCounts() {
        const text = textarea.value;
        charCount.textContent = text.length.toLocaleString();
        
        const words = text.trim().split(/\s+/).filter(w => w.length > 0);
        wordCount.textContent = words.length.toLocaleString();
    }

    textarea.addEventListener('input', updateCounts);
    // Initial run
    updateCounts();
}

/**
 * Handles scan form submission, shows loading scanner beam overlay.
 */
function initScanFormSubmission() {
    const form = document.getElementById('scan-form');
    const loadingOverlay = document.getElementById('scanner-loading');

    if (!form || !loadingOverlay) return;

    form.addEventListener('submit', (e) => {
        const textarea = document.getElementById('message_text');
        if (!textarea || textarea.value.trim().length === 0) {
            e.preventDefault();
            alert('Please input email or message text to analyze.');
            textarea?.focus();
            return;
        }

        // Display animated scanner beam
        loadingOverlay.style.display = 'flex';
        
        // Disable submit button to prevent double submissions
        const submitBtn = document.getElementById('btn-submit-scan');
        if (submitBtn) {
            submitBtn.disabled = true;
            submitBtn.style.opacity = '0.7';
        }
    });
}

/**
 * Mobile navigation menu toggle.
 */
function initMobileNav() {
    const toggle = document.getElementById('mobile-nav-toggle');
    const nav = document.getElementById('cyber-nav-menu');

    if (!toggle || !nav) return;

    toggle.addEventListener('click', () => {
        nav.classList.toggle('show');
        const icon = toggle.querySelector('i');
        if (icon) {
            if (nav.classList.contains('show')) {
                icon.className = 'fa-solid fa-xmark';
            } else {
                icon.className = 'fa-solid fa-bars';
            }
        }
    });

    // Close when clicking outside
    document.addEventListener('click', (e) => {
        if (!nav.contains(e.target) && !toggle.contains(e.target) && nav.classList.contains('show')) {
            nav.classList.remove('show');
            const icon = toggle.querySelector('i');
            if (icon) icon.className = 'fa-solid fa-bars';
        }
    });
}

/**
 * Shortcut: Ctrl + Enter or Cmd + Enter to quickly submit scan.
 */
function initKeybindings() {
    const textarea = document.getElementById('message_text');
    const form = document.getElementById('scan-form');

    if (!textarea || !form) return;

    textarea.addEventListener('keydown', (e) => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            if (textarea.value.trim().length > 0) {
                form.requestSubmit();
            }
        }
    });
}
