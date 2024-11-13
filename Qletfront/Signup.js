const modal = document.getElementById("signupPopup");
const signupButton = document.getElementById("signupButton");
const closePopup = document.getElementById("closePopup");
const signupForm = document.getElementById("signupForm");

signupButton.onclick = function() {
    modal.style.display = "block";
}

closePopup.onclick = function() {
    modal.style.display = "none";
}

window.onclick = function(event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}

document.getElementById('consumerButton').onclick = function() {
    signupForm.style.display = "block"; // Show the form
    document.getElementById('signupForm').dataset.type = 'consumer'; // Set type to consumer
}

document.getElementById('businessButton').onclick = function() {
    signupForm.style.display = "block"; // Show the form
    document.getElementById('signupForm').dataset.type = 'business'; // Set type to business
}

document.getElementById('submitSignup').onclick = function() {
    const type = signupForm.dataset.type;
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;

    // Replace with your actual API endpoint
    const apiEndpoint = type === 'consumer' ? '/api/signup/consumer' : '/api/signup/business';

    fetch(apiEndpoint, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ name, email })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            alert('Signup successful!');
            modal.style.display = "none"; 
        } else {
            alert('Signup failed: ' + data.message);
        }
 })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred during signup. Please try again.');
    });
}