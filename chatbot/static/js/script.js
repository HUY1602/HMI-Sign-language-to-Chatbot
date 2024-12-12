async function sendMessage() {
    const userInput = document.getElementById('userInput');
    const chatBox = document.getElementById('chatBox');
    const message = userInput.value.trim();

    if (!message) {
        alert("Please enter a message!");
        return;
    }

    // Display user's message
    const userMessage = document.createElement('div');
    userMessage.textContent = `You: ${message}`;
    chatBox.appendChild(userMessage);

    // Send the message to the backend
    try {
        const response = await fetch('/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ message })
        });

        const data = await response.json();

        // Display bot's response
        const botMessage = document.createElement('div');
        botMessage.textContent = `Bot: ${data.response}`;
        chatBox.appendChild(botMessage);
    } catch (error) {
        const errorMessage = document.createElement('div');
        errorMessage.textContent = "Bot: Sorry, an error occurred.";
        chatBox.appendChild(errorMessage);
    }

    // Clear the input box
    userInput.value = "";
    chatBox.scrollTop = chatBox.scrollHeight;
}
