async function sendMessage() {
  const input = document.getElementById("userInput");
  const chatBox = document.getElementById("chatBox");

  const userText = input.value.trim();
  if (!userText) return;

  const userMessage = `<p class="user">🧑 You: ${userText}</p>`;
  chatBox.innerHTML += userMessage;
  input.value = "";

  // Call backend API here
  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: userText }),
  });
  const data = await res.json();

  const botMessage = `<p class="bot">🤖 AI: ${data.reply}</p>`;
  chatBox.innerHTML += botMessage;

  chatBox.scrollTop = chatBox.scrollHeight;
}
