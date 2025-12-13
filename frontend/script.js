// GSAP Animations
gsap.registerPlugin(ScrollTrigger);

// Navbar scroll effect
let lastScroll = 0;
window.addEventListener('scroll', () => {
    const navbar = document.querySelector('.navbar');
    const currentScroll = window.pageYOffset;
    
    if (currentScroll > 100) {
        navbar.style.background = 'rgba(0, 0, 0, 0.95)';
        navbar.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.5)';
    } else {
        navbar.style.background = 'rgba(0, 0, 0, 0.8)';
        navbar.style.boxShadow = 'none';
    }
    
    lastScroll = currentScroll;
});

// Update active nav link on scroll
const sections = document.querySelectorAll('section[id]');
window.addEventListener('scroll', () => {
    let current = '';
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.clientHeight;
        if (scrollY >= (sectionTop - 200)) {
            current = section.getAttribute('id');
        }
    });

    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('active');
        if (link.getAttribute('href') === `#${current}`) {
            link.classList.add('active');
        }
    });
});

// Smooth scroll for navigation links
document.querySelectorAll('.nav-link, .btn-primary, .btn-secondary').forEach(link => {
    link.addEventListener('click', (e) => {
        const href = link.getAttribute('href');
        if (href && href.startsWith('#')) {
            e.preventDefault();
            const targetId = href.substring(1);
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                const offset = targetId === 'home' ? 0 : 80;
                const targetPosition = targetSection.offsetTop - offset;
                window.scrollTo({
                    top: targetPosition,
                    behavior: 'smooth'
                });
                
                // Update active nav link
                document.querySelectorAll('.nav-link').forEach(navLink => {
                    navLink.classList.remove('active');
                });
                if (link.classList.contains('nav-link')) {
                    link.classList.add('active');
                }
            }
        }
    });
});

// Hero section animations
gsap.from('.hero-title', {
    duration: 1,
    y: 50,
    opacity: 0,
    ease: 'power3.out'
});

gsap.from('.hero-subtitle', {
    duration: 1,
    y: 30,
    opacity: 0,
    delay: 0.2,
    ease: 'power3.out'
});

gsap.from('.stat-item', {
    duration: 0.8,
    y: 30,
    opacity: 0,
    stagger: 0.1,
    delay: 0.4,
    ease: 'back.out(1.7)'
});

gsap.from('.hero-buttons', {
    duration: 1,
    y: 30,
    opacity: 0,
    delay: 0.6,
    ease: 'power3.out'
});

// Pulse circles animation
gsap.from('.pulse-circle', {
    duration: 1.5,
    scale: 0,
    opacity: 0,
    ease: 'elastic.out(1, 0.5)'
});

// About section animations
gsap.utils.toArray('.model-card').forEach((card, index) => {
    gsap.from(card, {
        scrollTrigger: {
            trigger: card,
            start: 'top 85%',
            toggleActions: 'play none none reverse'
        },
        duration: 0.8,
        y: 50,
        opacity: 0,
        delay: index * 0.15,
        ease: 'power3.out'
    });
});

gsap.from('.section-header', {
    scrollTrigger: {
        trigger: '.about-section',
        start: 'top 80%',
        toggleActions: 'play none none reverse'
    },
    duration: 1,
    y: 50,
    opacity: 0,
    ease: 'power3.out'
});

gsap.from('.tech-stack', {
    scrollTrigger: {
        trigger: '.tech-stack',
        start: 'top 80%',
        toggleActions: 'play none none reverse'
    },
    duration: 1,
    y: 50,
    opacity: 0,
    ease: 'power3.out'
});

// Assessment section animations
gsap.from('.assessment-card', {
    scrollTrigger: {
        trigger: '.assessment-section',
        start: 'top 80%',
        toggleActions: 'play none none reverse'
    },
    duration: 1,
    y: 50,
    opacity: 0,
    scale: 0.95,
    ease: 'power3.out'
});

// Form submission
document.getElementById('predict-form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const resultDiv = document.getElementById('result');
    const payload = {
        gender: document.getElementById('gender').value,
        age: document.getElementById('age').value,
        hypertension: document.getElementById('hypertension').value,
        heart_disease: document.getElementById('heart_disease').value,
        smoking_history: document.getElementById('smoking_history').value,
        bmi: document.getElementById('bmi').value,
        hba1c_level: document.getElementById('hba1c').value,
        blood_glucose_level: document.getElementById('glucose').value
    };

    resultDiv.innerHTML = '<div style="text-align: center; color: #10b981; padding: 20px; font-size: 1.1rem;">🔬 Analyzing data...</div>';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        const data = await response.json();
        
        if (!response.ok || data.error) {
            resultDiv.innerHTML = `<div class="result-card result-high-risk">
                <div class="result-icon">❌</div>
                <div class="result-status">Error</div>
                <p style="color: #9ca3af;">${data.error || 'Unknown error occurred'}</p>
            </div>`;
        } else {
            const isHighRisk = data.prediction === 1;
            const probability = (data.probability * 100).toFixed(1);
            const riskClass = isHighRisk ? 'result-high-risk' : 'result-low-risk';
            const icon = isHighRisk ? '⚠️' : '✅';
            const status = isHighRisk ? 'High Risk Detected' : 'Low Risk';
            const advice = isHighRisk 
                ? '🩺 We strongly recommend scheduling a consultation with your healthcare provider for further evaluation and testing.'
                : '👍 Great! Continue maintaining a healthy lifestyle with regular exercise, balanced diet, and routine check-ups.';
            
            resultDiv.innerHTML = `<div class="result-card ${riskClass}">
                <div class="result-icon">${icon}</div>
                <div class="result-status">${status}</div>
                <div class="result-percentage">${probability}%</div>
                <div class="result-advice">${advice}</div>
                <button class="clear-btn" onclick="clearForm()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M3 6h18M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                    Clear Form
                </button>
            </div>`;
            
            // Animate the result with a slight delay
            setTimeout(() => {
                const resultCard = document.querySelector('.result-card');
                if (resultCard) {
                    gsap.fromTo(resultCard, 
                        {
                            y: 30,
                            opacity: 0,
                            scale: 0.9
                        },
                        {
                            duration: 0.8,
                            y: 0,
                            opacity: 1,
                            scale: 1,
                            ease: 'back.out(1.7)'
                        }
                    );
                }
            }, 50);
        }
    } catch (err) {
        resultDiv.innerHTML = `<div class="result-card result-high-risk">
            <div class="result-icon">❌</div>
            <div class="result-status">Connection Error</div>
            <p style="color: #9ca3af;">Unable to connect to the server. Please try again.</p>
        </div>`;
    }
});

// Clear form function
function clearForm() {
    document.getElementById('predict-form').reset();
    document.getElementById('result').innerHTML = '';
    
    // Smooth scroll to form
    const formSection = document.getElementById('assessment');
    const offset = 80;
    const targetPosition = formSection.offsetTop - offset;
    window.scrollTo({
        top: targetPosition,
        behavior: 'smooth'
    });
}
