async function sendMessage() {
  const input = document.getElementById("userInput");
  const chatBox = document.getElementById("chatBox");

  const userText = input.value.trim();
  if (!userText) return;

  const userMessage = `<p class="user">${userText}</p>`;
  chatBox.innerHTML += userMessage;
  input.value = "";

  const res = await fetch("/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: userText }),
  });
  const data = await res.json();

  const botMessage = `<p class="bot"><span class="bot-avatar">🤖</span>${data.reply}</p>`;
  chatBox.innerHTML += botMessage;
  chatBox.scrollTop = chatBox.scrollHeight;
}

// Theme toggle
const toggle = document.getElementById("themeToggle");
toggle.addEventListener("click", () => {
  document.body.classList.toggle("dark");
  toggle.textContent = document.body.classList.contains("dark") ? "☀️" : "🌙";
});
