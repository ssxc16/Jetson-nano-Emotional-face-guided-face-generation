document.addEventListener('DOMContentLoaded', function () {
    // Tab switching
    const tabs = document.querySelectorAll('.tab');
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            const targetPage = tab.dataset.page;

            // Update tabs
            tabs.forEach(t => t.classList.remove('active'));
            tab.classList.add('active');

            // Update pages
            document.querySelectorAll('.page').forEach(page => {
                page.classList.remove('active');
            });
            document.getElementById(`${targetPage}-page`).classList.add('active');
        });
    });

    function postAndLog(url, type, value) {
        const data = new FormData();
        data.append(type, value);

        fetch(url, {
            method: 'POST',
            body: data
        }).catch((error) => {
            console.error('Error:', error);
        });
    }

    // Seed input
    const seedInput = document.getElementById('seed-input');
    document.getElementById('seed-btn').addEventListener('click', () => {
        const seedValue = seedInput.value;
        postAndLog('/update_style', 'seed', seedValue);
    });

    // Sliders
    const hairSlider = document.getElementById('hair-slider');
    const beardSlider = document.getElementById('beard-slider');
    const smileSlider = document.getElementById('smile-slider');
    let lastAgeSliderPost = Date.now();
    let lastHairColorSliderPost = Date.now();
    let lastSmileSliderPost = Date.now();

    hairSlider.addEventListener('input', () => {
        if (Date.now() - lastAgeSliderPost > 150) {
            lastAgeSliderPost = Date.now();
            postAndLog('/update_style', 'hair', hairSlider.value / 100.0);
        }
    });

    beardSlider.addEventListener('input', () => {
        if (Date.now() - lastHairColorSliderPost > 150) {
            lastHairColorSliderPost = Date.now();
            postAndLog('/update_style', 'beard', beardSlider.value / 100.0);
        }
    });

    smileSlider.addEventListener('input', () => {
        if (Date.now() - lastSmileSliderPost > 150) {
            lastSmileSliderPost = Date.now();
            postAndLog('/update_style', 'smile', beardSlider.value / 100.0);
        }
    });

    // Capture and save_w button
    const capturedImage = document.getElementById('captured-image');
    const captureBtn = document.getElementById('capture-btn');
    const saveWBtn = document.getElementById('save-w-btn');
    const styleItems = Array.from(document.querySelectorAll('.style-item'),
        (item, index) => [
            item.querySelector('.style-source'),
            item.querySelector('.style-result')
        ]
    );

    // Initial state
    fetch('/bnt_state').then((response) => response.json()).then((data) => {
        saveWBtn.disabled = data.save_w_disabled;
    }).catch(
        error => console.error('Error:', error)
    );
    styleItems.forEach((item, index) => {
        item[0].src = `style${index}_in?t=${new Date().getTime()}`;
        item[1].src = `style${index}_out?t=${new Date().getTime()}`;
    });

    captureBtn.addEventListener('click', async () => {
        captureBtn.disabled = true;
        try {
            const response = await fetch('/capture');
            const data = await response.json();

            capturedImage.src = `/captured_image?t=${new Date().getTime()}`;
            if (data.success) {
                const styleResponse = await fetch('/generate_styles');

                if (styleResponse.ok) {
                    styleItems.forEach((result, index) => {
                        result[1].src = `style${index}_out?t=${new Date().getTime()}`;
                    });
                    saveWBtn.disabled = false;
                }
            }
        } catch (err) {
            console.error(`Capture error: ${err}`);
        }
        captureBtn.disabled = false;
    });

    saveWBtn.addEventListener('click', async () => {
        saveWBtn.disabled = true;
        try {
            await fetch('/save_w');
        } catch (err) {
            console.error('Save W error:', err);
        }
        saveWBtn.disabled = false;
    });
});
