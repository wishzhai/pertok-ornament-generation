/**
 * Ornament Generation Demo - Frontend Interaction Script
 */

document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const uploadForm = document.getElementById('upload-form');
    const midiFileInput = document.getElementById('midi-file');
    const temperatureSlider = document.getElementById('temperature');
    const topKSlider = document.getElementById('top-k');
    const topPSlider = document.getElementById('top-p');
    const tempValue = document.getElementById('temp-value');
    const topKValue = document.getElementById('top-k-value');
    const topPValue = document.getElementById('top-p-value');
    const generateBtn = document.getElementById('generate-btn');
    const resultsSection = document.getElementById('results-section');
    
    // Current uploaded filename
    let currentFilename = null;
    
    // MIDI visualization data
    let inputMidiData = null;
    let outputMidiData = null;
    
    // Update slider value display
    temperatureSlider.addEventListener('input', function() {
        tempValue.textContent = this.value;
    });
    
    topKSlider.addEventListener('input', function() {
        topKValue.textContent = this.value;
    });
    
    topPSlider.addEventListener('input', function() {
        topPValue.textContent = this.value;
    });
    
    // Handle form submission
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        // Check if file is selected
        if (!midiFileInput.files || midiFileInput.files.length === 0) {
            showAlert('Please select a MIDI file', 'danger');
            return;
        }
        
        // Disable button, show loading state
        generateBtn.disabled = true;
        generateBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
        
        try {
            // If file not uploaded yet, upload first
            if (!currentFilename) {
                const uploadResult = await uploadFile(midiFileInput.files[0]);
                if (!uploadResult.success) {
                    throw new Error(uploadResult.error || 'File upload failed');
                }
                currentFilename = uploadResult.filename;
            }
            
            // Generate ornaments
            const generateResult = await generateOrnaments(currentFilename, {
                temperature: temperatureSlider.value,
                top_k: topKSlider.value,
                top_p: topPSlider.value
            });
            
            // Display results
            displayResults(generateResult);
            
        } catch (error) {
            showAlert(error.message, 'danger');
        } finally {
            // Restore button state
            generateBtn.disabled = false;
            generateBtn.textContent = 'Generate Ornaments';
        }
    });
    
    // Reset current filename when selecting a new file
    midiFileInput.addEventListener('change', function() {
        currentFilename = null;
    });
    
    /**
     * Upload MIDI file
     * @param {File} file - MIDI file
     * @returns {Promise<Object>} Upload result
     */
    async function uploadFile(file) {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });
        
        return await response.json();
    }
    
    /**
     * Generate ornaments
     * @param {string} filename - Uploaded filename
     * @param {Object} params - Generation parameters
     * @returns {Promise<Object>} Generation result
     */
    async function generateOrnaments(filename, params) {
        const response = await fetch('/generate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                filename: filename,
                temperature: params.temperature,
                top_k: params.top_k,
                top_p: params.top_p
            })
        });
        
        return await response.json();
    }
    
    /**
     * Display generation results
     * @param {Object} result - Generation result
     */
    function displayResults(result) {
        if (!result.success) {
            showAlert(result.error || 'Generation failed', 'danger');
            return;
        }
        
        // Show results section
        resultsSection.style.display = 'block';
        
        // Scroll to results section
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
        // Load and display MusicXML scores using OpenSheetMusicDisplay
        loadMusicXMLScore('input-score', `/scores/${result.input_score}`);
        loadMusicXMLScore('output-score', `/scores/${result.output_score}`);
        
        // Set up MIDI players
        const inputPlayer = document.getElementById('input-player');
        const outputPlayer = document.getElementById('output-player');
        inputPlayer.src = `/uploads/${result.input_midi}`;
        outputPlayer.src = `/results/${result.output_midi}`;
        
        // Set download links
        const inputDownload = document.getElementById('input-download');
        const outputDownload = document.getElementById('output-download');
        inputDownload.href = `/uploads/${result.input_midi}`;
        outputDownload.href = `/results/${result.output_midi}`;
        
        // Display analysis results
        displayAnalysis(result.analysis);
    }
    
    /**
     * Load and display MusicXML score using OpenSheetMusicDisplay
     * @param {string} containerId - ID of container element
     * @param {string} musicXmlUrl - URL of MusicXML file
     */
    async function loadMusicXMLScore(containerId, musicXmlUrl) {
        try {
            const container = document.getElementById(containerId);
            if (!container) {
                console.error(`Container ${containerId} not found`);
                return;
            }
            
            // Clear container
            container.innerHTML = '';
            
            // Create OSMD instance with different settings based on container ID
            const options = {
                autoResize: true,
                backend: 'svg',
                drawTitle: false,
                drawSubtitle: false,
                drawComposer: false,
                drawCredits: false,
                // Only use colors defined in MusicXML (e.g., purple for ornaments). No auto multi-coloring.
                coloringMode: 0, // XML
                coloringEnabled: true
            };
            
            // Let the backend handle coloring - don't set any default colors in frontend
            // This ensures original notes are black and only ornaments are purple
            // The coloring is already handled in the backend midi_to_score function
            
            const osmd = new opensheetmusicdisplay.OpenSheetMusicDisplay(container, options);
            
            // Fetch and load MusicXML (cache-busting to ensure latest)
            const cacheBustedUrl = musicXmlUrl + (musicXmlUrl.includes('?') ? '&' : '?') + 'v=' + Date.now();
            const response = await fetch(cacheBustedUrl, { cache: 'no-store' });
            if (!response.ok) {
                throw new Error(`Failed to fetch MusicXML: ${response.status}`);
            }
            
            const musicXmlText = await response.text();
            await osmd.load(musicXmlText);
            osmd.render();
            
            console.log(`Successfully loaded MusicXML for ${containerId}`);
        } catch (error) {
            console.error(`Error loading MusicXML for ${containerId}:`, error);
            // Show error message in container
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = `<div class="alert alert-warning">Failed to load score: ${error.message}</div>`;
            }
        }
    }
    
    /**
     * Load MIDI files for visualization
     * @param {string} inputMidiUrl - URL of input MIDI file
     * @param {string} outputMidiUrl - URL of output MIDI file with ornaments
     */
    async function loadMidiForVisualization(inputMidiUrl, outputMidiUrl) {
        try {
            // Create piano roll visualizations when audio players are played
            const inputAudio = document.getElementById('input-audio');
            const outputAudio = document.getElementById('output-audio');
            
            // Set up visualization containers if they don't exist
            if (!document.getElementById('input-piano-roll')) {
                const inputScoreContainer = document.querySelector('#input-score').parentElement;
                const inputVisualization = document.createElement('div');
                inputVisualization.id = 'input-piano-roll';
                inputVisualization.className = 'piano-roll mt-3';
                inputScoreContainer.appendChild(inputVisualization);
            }
            
            if (!document.getElementById('output-piano-roll')) {
                const outputScoreContainer = document.querySelector('#output-score').parentElement;
                const outputVisualization = document.createElement('div');
                outputVisualization.id = 'output-piano-roll';
                outputVisualization.className = 'piano-roll mt-3';
                outputScoreContainer.appendChild(outputVisualization);
            }
            
            // Set up event listeners for audio playback to update visualization
            inputAudio.addEventListener('play', function() {
                // Start visualization for input MIDI
                visualizeMidi('input-piano-roll', inputMidiUrl, this);
            });
            
            outputAudio.addEventListener('play', function() {
                // Start visualization for output MIDI with ornaments
                visualizeMidi('output-piano-roll', outputMidiUrl, this, true);
            });
        } catch (error) {
            console.error('Error loading MIDI for visualization:', error);
        }
    }
    
    /**
     * Visualize MIDI as piano roll during playback
     * @param {string} containerId - ID of container element
     * @param {string} midiUrl - URL of MIDI file
     * @param {HTMLAudioElement} audioElement - Audio element for playback
     * @param {boolean} highlightOrnaments - Whether to highlight ornaments
     */
    function visualizeMidi(containerId, midiUrl, audioElement, highlightOrnaments = false) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        // Clear previous visualization
        container.innerHTML = '';
        
        // Create canvas for piano roll
        const canvas = document.createElement('canvas');
        canvas.width = container.clientWidth;
        canvas.height = 200;
        container.appendChild(canvas);
        
        const ctx = canvas.getContext('2d');
        
        // Fetch MIDI file and parse
        fetch(midiUrl)
            .then(response => response.arrayBuffer())
            .then(buffer => {
                // Here we would normally parse the MIDI file
                // For this demo, we'll create a simple visualization with mock data
                const mockNotes = generateMockNotes(highlightOrnaments);
                
                // Start visualization
                let startTime = Date.now();
                let animationFrame;
                
                function drawPianoRoll() {
                    const currentTime = (Date.now() - startTime) / 1000;
                    const duration = audioElement.duration || 60; // Default to 60 seconds if duration unknown
                    
                    // Clear canvas
                    ctx.clearRect(0, 0, canvas.width, canvas.height);
                    
                    // Draw time marker
                    const timePosition = (currentTime / duration) * canvas.width;
                    ctx.fillStyle = '#333';
                    ctx.fillRect(timePosition, 0, 2, canvas.height);
                    
                    // Draw notes
                    mockNotes.forEach(note => {
                        const noteStart = (note.start / duration) * canvas.width;
                        const noteEnd = (note.end / duration) * canvas.width;
                        const noteWidth = noteEnd - noteStart;
                        
                        // Calculate vertical position (higher pitch = higher position)
                        const noteHeight = 5;
                        const noteY = canvas.height - (note.pitch - 21) * noteHeight;
                        
                        // Use consistent colors with the MusicXML display
                        ctx.fillStyle = note.isOrnament ? '#9C27B0' : '#000000';
                        
                        // Draw note rectangle
                        ctx.fillRect(noteStart, noteY, noteWidth, noteHeight);
                    });
                    
                    // Continue animation if audio is playing
                    if (!audioElement.paused) {
                        animationFrame = requestAnimationFrame(drawPianoRoll);
                    }
                }
                
                // Start animation
                animationFrame = requestAnimationFrame(drawPianoRoll);
                
                // Stop animation when audio ends or is paused
                audioElement.addEventListener('pause', () => {
                    cancelAnimationFrame(animationFrame);
                });
                
                audioElement.addEventListener('ended', () => {
                    cancelAnimationFrame(animationFrame);
                });
            })
            .catch(error => {
                console.error('Error loading MIDI file:', error);
                container.innerHTML = '<div class="alert alert-warning">Failed to load MIDI visualization</div>';
            });
    }
    
    /**
     * Generate mock notes for visualization
     * @param {boolean} includeOrnaments - Whether to include ornaments
     * @returns {Array} Array of note objects
     */
    function generateMockNotes(includeOrnaments = false) {
        const notes = [];
        const duration = 60; // 60 seconds total duration
        
        // Generate main notes (every 2 seconds)
        for (let i = 0; i < 30; i++) {
            const start = i * 2;
            const end = start + 1.5;
            const pitch = 60 + Math.floor(Math.random() * 24); // C4 to C6
            
            notes.push({
                start,
                end,
                pitch,
                isOrnament: false
            });
        }
        
        // Add ornaments if requested
        if (includeOrnaments) {
            for (let i = 0; i < 20; i++) {
                const start = i * 3 + 0.5;
                const end = start + 0.3;
                const pitch = 72 + Math.floor(Math.random() * 12); // Higher pitches for ornaments
                
                notes.push({
                    start,
                    end,
                    pitch,
                    isOrnament: true
                });
            }
        }
        
        return notes;
    }
    
    /**
     * Display ornament analysis
     * @param {Object} analysis - Analysis results
     */
    function displayAnalysis(analysis) {
        const analysisContent = document.getElementById('analysis-content');
        
        if (!analysis) {
            analysisContent.innerHTML = '<div class="alert alert-warning">No analysis data available</div>';
            return;
        }
        
        let html = '';
        
        // Ornament statistics
        html += `
            <div class="analysis-item">
                <div class="analysis-title">Ornament Statistics</div>
                <div class="row">
                    <div class="col-md-6">
                        <p>Original Notes: <strong>${analysis.original_notes || 0}</strong></p>
                        <p>Ornament Notes: <strong class="ornament-note">${analysis.ornament_notes || 0}</strong></p>
                    </div>
                    <div class="col-md-6">
                        <p>Ornament Density: <strong>${analysis.ornament_density ? (analysis.ornament_density * 100).toFixed(2) + '%' : '0%'}</strong></p>
                        <p>Microtiming Adjustments: <strong>${analysis.microtiming_adjustments || 0}</strong></p>
                    </div>
                </div>
            </div>
        `;
        
        // Ornament type distribution
        if (analysis.ornament_types) {
            html += `
                <div class="analysis-item">
                    <div class="analysis-title">Ornament Type Distribution</div>
                    <div class="row">
            `;
            
            for (const [type, count] of Object.entries(analysis.ornament_types)) {
                html += `
                    <div class="col-md-4 col-6">
                        <p>${type}: <strong>${count}</strong></p>
                    </div>
                `;
            }
            
            html += `
                    </div>
                </div>
            `;
        }
        
        analysisContent.innerHTML = html;
    }
    
    /**
     * Show alert message
     * @param {string} message - Alert message
     * @param {string} type - Alert type (success, danger, warning, info)
     */
    function showAlert(message, type = 'info') {
        // Create alert element
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.role = 'alert';
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        // Add to page
        const container = document.querySelector('.container');
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto-close after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 150);
        }, 5000);
    }
});