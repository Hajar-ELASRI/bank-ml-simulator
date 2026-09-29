const form = document.querySelector("#prediction-form");
const fieldsContainer = document.querySelector("#fields");
const submitButton = document.querySelector("#submit-button");
const errorMessage = document.querySelector("#error-message");
const resultCard = document.querySelector("#result-card");

const labels = {
  age: "Âge", job: "Profession", marital: "Situation familiale", education: "Éducation",
  default: "Défaut de crédit", balance: "Solde", housing: "Prêt immobilier", loan: "Prêt personnel",
  contact: "Type de contact", day: "Jour", month: "Mois", duration: "Durée (secondes)",
  campaign: "Nombre de contacts", pdays: "Jours depuis le dernier contact", previous: "Contacts précédents",
  poutcome: "Résultat précédent"
};

const algorithmNames = { svm: "SVM", random_forest: "Random Forest", mlp: "MLP" };

function createField(name, schema) {
  const wrapper = document.createElement("div");
  wrapper.className = "field";
  const label = document.createElement("label");
  label.htmlFor = name;
  label.textContent = labels[name] || name;

  let input;
  if (schema.categorical_features[name]) {
    input = document.createElement("select");
    schema.categorical_features[name].forEach((value) => {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      input.appendChild(option);
    });
  } else {
    const settings = schema.numeric_features[name];
    input = document.createElement("input");
    input.type = "number";
    input.min = settings.min;
    input.max = settings.max;
    input.step = settings.integer ? "1" : "any";
    input.value = settings.default;
  }
  input.id = name;
  input.name = name;
  input.required = true;
  wrapper.append(label, input);
  return wrapper;
}

async function initializeForm() {
  try {
    const response = await fetch("/api/schema");
    if (!response.ok) throw new Error("Impossible de charger le schéma des données.");
    const schema = await response.json();
    schema.feature_order.forEach((name) => fieldsContainer.appendChild(createField(name, schema)));
    document.querySelector("#dataset-info").textContent = `${schema.dataset_rows.toLocaleString("fr-FR")} observations · ${schema.feature_order.length} variables`;
  } catch (error) {
    showError(error.message);
    submitButton.disabled = true;
  }
}

function showError(message) {
  errorMessage.textContent = message;
  errorMessage.hidden = false;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  resultCard.hidden = true;
  submitButton.disabled = true;
  submitButton.firstElementChild.textContent = "CALCUL EN COURS…";

  const values = Object.fromEntries(new FormData(form).entries());
  form.querySelectorAll('input[type="number"]').forEach((input) => {
    values[input.name] = Number(input.value);
  });

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values)
    });
    const data = await response.json();
    if (!response.ok) {
      const detail = Array.isArray(data.detail) ? data.detail.map((item) => item.msg).join(", ") : data.detail;
      throw new Error(detail || "La prédiction a échoué.");
    }

    const prediction = document.querySelector("#prediction");
    prediction.textContent = data.prediction === "yes" ? "OUI" : "NON";
    prediction.classList.toggle("no", data.prediction !== "yes");
    document.querySelector("#result-algorithm").textContent = algorithmNames[data.algorithm];
    document.querySelector("#result-accuracy").textContent = `${(data.accuracy * 100).toFixed(2)} %`;
    resultCard.hidden = false;
    resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (error) {
    showError(error.message);
  } finally {
    submitButton.disabled = false;
    submitButton.firstElementChild.textContent = "SIMULER";
  }
});

initializeForm();
