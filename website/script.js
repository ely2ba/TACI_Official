// TACI - Elite Neural Network Animation
class TACINeuralNetwork {
    constructor() {
        this.nodes = document.querySelectorAll('.node');
        this.inputNodes = document.querySelectorAll('.input-node');
        this.hiddenNodes = document.querySelectorAll('.hidden-node');
        this.outputNodes = document.querySelectorAll('.output-node');
        this.inputLabels = document.querySelectorAll('.input-label');
        this.outputLabels = document.querySelectorAll('.output-label');
        this.progressRing = document.querySelector('.progress-ring');
        this.percentageText = document.querySelector('.percentage');
        this.connectionsContainer = document.querySelector('.connections');
        
        this.targetPercentage = 72;
        this.currentProgress = 0;
        this.isRunning = false;
        this.currentOccupationSet = 0;
        
        // Three sets of occupations to cycle through
        this.occupationSets = [
            {
                // Set 1: Professional Services
                occupations: [
                    "Research Scientist",
                    "Software Engineer", 
                    "Financial Analyst",
                    "Physician",
                    "Lawyer",
                    "Creative Director",
                    "Business Strategist"
                ]
            },
            {
                // Set 2: Creative & Media
                occupations: [
                    "Graphic Designer",
                    "Marketing Manager",
                    "Data Scientist",
                    "Architect",
                    "Journalist",
                    "UX Designer",
                    "Product Manager"
                ]
            },
            {
                // Set 3: Technical & Operations
                occupations: [
                    "Cybersecurity Analyst",
                    "Operations Manager",
                    "Investment Banker",
                    "Clinical Researcher",
                    "Content Strategist",
                    "Engineering Manager",
                    "Consulting Director"
                ]
            }
        ];
        
        this.init();
    }

    init() {
        this.generateConnections();
        setTimeout(() => {
            this.startSequence();
        }, 2000);
    }

    generateConnections() {
        const connections = [];
        
        // Input to first hidden layer
        this.inputNodes.forEach((inputNode, i) => {
            document.querySelectorAll('.hidden-layer-1 .node').forEach((hiddenNode, j) => {
                connections.push(this.createConnection(inputNode, hiddenNode));
            });
        });

        // Hidden layer 1 to 2
        document.querySelectorAll('.hidden-layer-1 .node').forEach((node1) => {
            document.querySelectorAll('.hidden-layer-2 .node').forEach((node2) => {
                connections.push(this.createConnection(node1, node2));
            });
        });

        // Hidden layer 2 to 3
        document.querySelectorAll('.hidden-layer-2 .node').forEach((node2) => {
            document.querySelectorAll('.hidden-layer-3 .node').forEach((node3) => {
                connections.push(this.createConnection(node2, node3));
            });
        });

        // Hidden layer 3 to output
        document.querySelectorAll('.hidden-layer-3 .node').forEach((hiddenNode) => {
            this.outputNodes.forEach((outputNode) => {
                connections.push(this.createConnection(hiddenNode, outputNode));
            });
        });

        connections.forEach(conn => {
            this.connectionsContainer.appendChild(conn);
        });
    }

    createConnection(node1, node2) {
        const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        const x1 = node1.getAttribute('cx');
        const y1 = node1.getAttribute('cy');
        const x2 = node2.getAttribute('cx');
        const y2 = node2.getAttribute('cy');
        
        line.setAttribute('x1', x1);
        line.setAttribute('y1', y1);
        line.setAttribute('x2', x2);
        line.setAttribute('y2', y2);
        line.setAttribute('class', 'connection');
        
        return line;
    }

    startSequence() {
        this.animateProgressRing();
        this.updateOccupationLabels();
        this.runNeuralFlow();
        
        // Loop every 8 seconds to match neural flow
        setInterval(() => {
            this.runNeuralFlow();
        }, 8000);
    }

    updateOccupationLabels() {
        const currentSet = this.occupationSets[this.currentOccupationSet];
        this.inputLabels.forEach((label, index) => {
            if (currentSet.occupations[index]) {
                label.textContent = currentSet.occupations[index];
            }
        });
    }

    cycleToNextOccupationSet() {
        // Faster fade out
        this.inputLabels.forEach(label => {
            label.style.transition = 'opacity 0.2s ease';
            label.style.opacity = '0';
        });

        // Update to next set immediately after fade
        setTimeout(() => {
            this.currentOccupationSet = (this.currentOccupationSet + 1) % this.occupationSets.length;
            this.updateOccupationLabels();
            
            // Faster fade in
            setTimeout(() => {
                this.inputLabels.forEach(label => {
                    label.style.opacity = '0.7';
                });
            }, 20);
        }, 200);
    }

    animateProgressRing() {
        const circumference = 2 * Math.PI * 45;
        const duration = 5000;
        const start = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - start;
            const progress = Math.min(elapsed / duration, 1);
            
            // Smooth easing
            const eased = 1 - Math.pow(1 - progress, 3);
            const value = Math.floor(eased * this.targetPercentage);
            const offset = circumference - (eased * this.targetPercentage / 100) * circumference;
            
            this.percentageText.textContent = value;
            this.progressRing.style.strokeDashoffset = offset;
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        
        requestAnimationFrame(animate);
    }

    runNeuralFlow() {
        if (this.isRunning) return;
        this.isRunning = true;
        
        // Reset all states
        this.resetAll();
        
        // Step 1: Activate input nodes and labels sequentially
        this.inputNodes.forEach((node, index) => {
            setTimeout(() => {
                this.activateNode(node);
                this.activateLabel(this.inputLabels[index]);
            }, index * 200);
        });
        
        // Step 2: Process through hidden layers
        setTimeout(() => this.activateHiddenLayer(1), 2000);
        setTimeout(() => this.activateHiddenLayer(2), 3000);
        setTimeout(() => this.activateHiddenLayer(3), 4000);
        
        // Step 3: Activate output
        setTimeout(() => {
            this.outputNodes.forEach((node, index) => {
                setTimeout(() => {
                    this.activateNode(node);
                    this.activateLabel(this.outputLabels[index]);
                }, index * 300);
            });
        }, 5000);
        
        // Step 4: Transition occupations after animation completes and before reset
        setTimeout(() => {
            this.cycleToNextOccupationSet();
        }, 7000);
        
        // Step 5: Reset for next cycle
        setTimeout(() => {
            this.resetAll();
            this.isRunning = false;
        }, 7500);
    }

    resetAll() {
        // Reset nodes
        this.nodes.forEach(node => {
            node.classList.remove('active');
        });
        
        // Reset labels (but don't reset input label text)
        [...this.inputLabels, ...this.outputLabels].forEach(label => {
            label.classList.remove('active');
        });
        
        // Reset connections
        document.querySelectorAll('.connection').forEach(conn => {
            conn.classList.remove('active');
        });
    }

    activateNode(node) {
        node.classList.add('active');
        
        // Activate connections from this node
        this.activateConnectionsFrom(node);
    }

    activateLabel(label) {
        if (label) {
            label.classList.add('active');
        }
    }

    activateHiddenLayer(layerNum) {
        const layerNodes = document.querySelectorAll(`.hidden-layer-${layerNum} .node`);
        layerNodes.forEach((node, index) => {
            setTimeout(() => {
                this.activateNode(node);
            }, index * 100);
        });
    }

    activateConnectionsFrom(sourceNode) {
        const connections = document.querySelectorAll('.connection');
        const sourceX = sourceNode.getAttribute('cx');
        const sourceY = sourceNode.getAttribute('cy');
        
        connections.forEach(conn => {
            const x1 = conn.getAttribute('x1');
            const y1 = conn.getAttribute('y1');
            
            if (x1 === sourceX && y1 === sourceY) {
                setTimeout(() => {
                    conn.classList.add('active');
                }, Math.random() * 200);
            }
        });
    }

}

// Enhanced interaction effects
class InteractionController {
    constructor() {
        this.addNodeInteractions();
        this.addLabelInteractions();
    }

    addNodeInteractions() {
        document.querySelectorAll('.node').forEach(node => {
            node.addEventListener('mouseenter', () => {
                if (!node.classList.contains('active')) {
                    node.style.transform = 'scale(1.6)';
                    node.style.filter = 'brightness(1.4) drop-shadow(0 0 10px rgba(255, 255, 255, 0.7))';
                }
            });

            node.addEventListener('mouseleave', () => {
                if (!node.classList.contains('active')) {
                    node.style.transform = '';
                    node.style.filter = '';
                }
            });

            node.addEventListener('click', () => {
                node.classList.add('active');
                setTimeout(() => {
                    node.classList.remove('active');
                }, 2000);
            });
        });
    }

    addLabelInteractions() {
        document.querySelectorAll('.label').forEach(label => {
            label.addEventListener('mouseenter', () => {
                if (!label.classList.contains('active')) {
                    label.style.color = '#CCCCCC';
                    label.style.transform = label.classList.contains('output-label') ? 
                        'translateX(-12px)' : 'translateX(12px)';
                }
            });

            label.addEventListener('mouseleave', () => {
                if (!label.classList.contains('active')) {
                    label.style.color = '';
                    label.style.transform = '';
                }
            });
        });
    }
}

// Initialize when DOM loads
document.addEventListener('DOMContentLoaded', () => {
    new TACINeuralNetwork();
    new InteractionController();
    
    // Add subtle scroll-based animations
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };
    
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.transform = 'translateY(0)';
                entry.target.style.opacity = '1';
            }
        });
    }, observerOptions);
    
    // Observe elements for scroll animations
    document.querySelectorAll('.brand, .neural-visualization, .content, footer').forEach(el => {
        observer.observe(el);
    });
});