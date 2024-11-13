document.getElementById('businessType').addEventListener('change', function() {
    const subBusinessType = document.getElementById('subBusinessType');
    const subType = document.getElementById('subType'); // Corrected the ID
    subType.innerHTML = '';

    if (this.value === 'medical') {
        subBusinessType.classList.remove('hidden');
        subType.innerHTML = `
            <option value="ayurvedic">Ayurvedic</option>
            <option value="allopathic">Allopathic</option>
            <option value="homeopathic">Homeopathic</option>
        `;
    } else if (this.value === 'restaurant') {
        subBusinessType.classList.remove('hidden');
        subType.innerHTML = `
            <option value="vegetarian">Vegetarian</option>
            <option value="vegan">Vegan</option>
            <option value="non-vegetarian">Non-Vegetarian</option>
        `;
    } else if (this.value === 'hotel') {
        subBusinessType.classList.remove('hidden');
        subType.innerHTML = `
            <option value="villas">Villas</option>
            <option value="hotels">Hotels</option>
        `;
    } else {
        subBusinessType.classList.add('hidden');
    }
});

document.getElementById('signupForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    const name = document.getElementById('name').value;
    const businessType = document.getElementById('businessType').value;
    const subType = document.getElementById('subType').value;
    const businessName = document.getElementById('businessName').value;
    const gst = document.getElementById('gst').value;
    const email = document.getElementById('email').value;
    const timing = document.getElementById('timing').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    // Basic validation for password match
    if (password !== confirmPassword) {
        alert('Passwords do not match!');
        return;
    }

    // Prepare data for API request
    const signupData = {
        name,
        businessType,
        subType,
        businessName,
        gst,
        email,
        timing,
        password
    };

    // Send data to the server
    fetch('/api/signup/business', { // Replace with your actual API endpoint
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(signupData)
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Sign up successful!');
        } else {
            alert('Sign up failed: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error during signup:', error);
        alert('An error occurred during signup. Please try again.');
    });
});