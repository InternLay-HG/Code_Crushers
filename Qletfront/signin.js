document.getElementById('signInForm').addEventListener('submit', function(event) {
    event.preventDefault(); 

    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;

    console.log('Email:', email);
    console.log('Password:', password);

    // Updated API endpoint
    const apiEndpoint = 'https://queletbackend.onrender.com/auth/customer-login';

    // Send a POST request to the API
    fetch(apiEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            email: email,
            password: password
        })
    })
    .then(response => {
        if (!response.ok) {
            return response.text().then(text => {
                throw new Error(`Network response was not ok: ${response.status} ${response.statusText}. Response body: ${text}`);
            });
        }
        return response.json();
    })
    .then(data => {
        console.log('Success:', data);
        
        // Assuming the token is in data.token
        const token = data.token; // Adjust this based on your API response structure

        // Store the token in localStorage or sessionStorage
        localStorage.setItem('authToken', token); // or sessionStorage.setItem('authToken', token);

        alert('Sign In Successful!');

        // Fetch protected resource after successful sign-in
        fetchProtectedResource();
    })
    .catch(error => {
        console.error('Error:', error);
        let userMessage = 'Sign In Failed: ' + error.message;

        // Customize the message based on the error
        if (error.message.includes('403')) {
            userMessage = 'Sign In Failed: You do not have permission to access this resource.';
        } else if (error.message.includes('401')) {
            userMessage = 'Sign In Failed: Incorrect email or password.';
        } else if (error.message.includes('400')) {
            userMessage = 'Sign In Failed: Bad request. Please check your input.';
        } else if (error.message.includes('404')) {
            userMessage = 'Sign In Failed: The requested resource was not found.';
        } else if (error.message.includes('500')) {
            userMessage = 'Sign In Failed: Internal server error. Please try again later.';
        }

        alert(userMessage);
    });

    // Reset the form after submission
    document.getElementById('signInForm').reset();
});

document.getElementById('signupLink').addEventListener('click', function() {
    window.location.href = 'signup.html'; 
});

// Function to fetch protected resource
function fetchProtectedResource() {
    const token = localStorage.getItem('authToken'); // or sessionStorage.getItem('authToken');

    fetch('https://queletbackend.onrender.com/protected-resource', {
        method: 'GET',
        headers: {
            'Authorization': `Bearer ${token}`, // Include the token in the Authorization header
            'Content-Type': 'application/json'
        }
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`Error: ${response.status} ${response.statusText}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Protected resource data:', data);
        // You can handle the protected resource data here (e.g., display it on the page)
    })
    .catch(error => {
        console.error('Error fetching protected resource:', error);
        alert('Failed to fetch protected resource: ' + error.message);
    });
}