document.getElementById('signupForm').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent form submission

    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
    const mobile = document.getElementById('mobile').value;

    if (password !== confirmPassword) {
        alert('Passwords do not match!');
        return;
    }

    const signupData = {
        name,
        email,
        password,
        mobile
    };

    const targetUrl = 'https://queletbackend.onrender.com/auth/customer-signup';

    fetch(targetUrl, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(signupData)
    })
    .then(response => {
        // Check if the response is OK
        if (!response.ok) {
            // Attempt to parse the response as text
            return response.text().then(text => {
                // Log the raw response for debugging
                console.error('Error response text:', text);
                // Create a custom error message
                throw new Error(text); // Use the raw text as the error message
            });
        }
        // If the response is OK, parse it as JSON
        return response.json();
    })
    .then(data => {
        console.log('Response data:', data);
        if (data.success) {
            alert('Sign up successful!');
            document.getElementById('signupForm').reset();
        } else {
            alert('Sign up failed: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error during signup:', error);
        // Display a user-friendly error message
        if (error.message.includes('username already taken')) {
            alert('Sign up failed: The username is already taken. Please choose a different one.');
        } else if (error.message.includes('email already taken')) {
            alert('Sign up failed: The email is already registered. Please use a different email.');
        } else {
            alert('An error occurred during signup: ' + error.message);
        }
    });
});