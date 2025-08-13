const sendBtn = document.getElementById("send-btn");
  const input = document.getElementById("user-input");
  const chatBox = document.getElementById("chat-box");

  function appendMessage(type, text) {
    const div = document.createElement("div");
    div.className = `message ${
      type === "user" ? "user-message" : "bot-message"
    }`;
    div.textContent = text;
    chatBox.appendChild(div);
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function sendMessage() {
    const message = input.value.trim();
    if (!message) return;
    appendMessage("user", message);
    input.value = "";

    fetch("/get_response", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    })
      .then((res) => res.json())
      .then((data) => appendMessage("bot", data.reply))
      .catch(() => appendMessage("bot", "⚠️ Error connecting to server."));
  }

  sendBtn.addEventListener("click", sendMessage);
  input.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });