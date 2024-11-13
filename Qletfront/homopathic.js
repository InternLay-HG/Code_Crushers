document.addEventListener("DOMContentLoaded", function() {
    // Fetch available slots from the API
    fetch('/api/available-slots')
        .then(response => response.json())
        .then(data => {
            document.getElementById('available-slots').innerText = data.availableSlots;
        })
        .catch(error => {
            console.error('Error fetching available slots:', error);
            document.getElementById('available-slots').innerText = 'Error loading slots';
        });

    // Register button event listener
    document.getElementById('register-button').addEventListener('click', function() {
        fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body : JSON.stringify({ eventId: '12345' }) // Replace with actual event ID
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Registration successful!');
            } else {
                alert('Registration failed: ' + data.message);
            }
        })
        .catch(error => {
            console.error('Error during registration:', error);
            alert('An error occurred while registering.');
        });
    });

    // Search button event listener
    document.getElementById('search-button').addEventListener('click', function() {
        const query = document.getElementById('search-input').value;
        // Implement search functionality here
        console.log('Searching for:', query);
    });
});