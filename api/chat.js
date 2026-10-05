// Petit serveur : il reçoit les messages, appelle l'IA avec TA clé secrète,
// puis renvoie la réponse. La clé n'est jamais visible par les visiteurs.

const MODEL = "claude-sonnet-5-5";

// 👉 C'est ici que tu définis la personnalité de ton agent.
const SYSTEM_PROMPT = `Tu es un assistant IA francophone, amical et clair.
Tu réponds de façon simple et concrète, avec des exemples adaptés au contexte africain.
Si tu ne sais pas quelque chose, tu le dis honnêtement.`;

module.exports = async (req, res) => {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Méthode non autorisée" });
  }

  const { messages } = req.body || {};
  if (!Array.isArray(messages) || messages.length === 0) {
    return res.status(400).json({ error: "Aucun message reçu" });
  }

  try {
    const r = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-api-key": process.env.ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01",
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 1000,
        system: SYSTEM_PROMPT,
        messages: messages.slice(-20), // garde les 20 derniers messages
      }),
    });

    const data = await r.json();
    if (!r.ok) {
      return res
        .status(r.status)
        .json({ error: data.error?.message || "Erreur du service IA" });
    }

    const reply = data.content
      .filter((b) => b.type === "text")
      .map((b) => b.text)
      .join("\n");

    return res.status(200).json({ reply });
  } catch (e) {
    return res.status(500).json({ error: "Erreur serveur" });
  }
};
