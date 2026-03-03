document.addEventListener("DOMContentLoaded", function () {

    const stored = sessionStorage.getItem("fareResults");

    if (!stored) {
        window.location.href = "index.html";
        return;
    }

    const data = JSON.parse(stored);

    const platforms = [
        {
            name: "inDrive",
            price: data.indrive_price,
            logo: "indrive.png",
            url: "https://indrive.com/"
        },
        {
            name: "Ola",
            price: data.ola_price,
            logo: "ola.png",
            url: "https://www.olacabs.com/"
        },
        {
            name: "Uber",
            price: data.uber_price,
            logo: "uber.png",
            url: "https://www.uber.com/in/en/start-riding/"
        },
        {
            name: "Rapido",
            price: data.rapido_price,
            logo: "rapido.png",
            url: "https://www.rapido.bike/Home"
        }
    ];

    const validPlatforms = platforms.map(p => ({
        ...p,
        price: Number(p.price)
    }));

    console.log("Stored Data:", data);
console.log("Uber:", data.uber_price);
console.log("Rapido:", data.rapido_price);

    validPlatforms.sort((a, b) => a.price - b.price);

    const container = document.getElementById("rcResultsContainer");
    container.innerHTML = "";

    console.log("---- Prediction Debug Info ----");
    console.log("Hour:", data.computed_hour);
    console.log("Traffic:", data.computed_traffic);
    console.log("Surge:", data.computed_surge);
    console.log("Cache Hit:", data.cache_hit);
    console.log("--------------------------------");

    validPlatforms.forEach((platform, index) => {

        const div = document.createElement("div");
        div.className = index === 0
            ? "rc-list-item rc-list-item--highlight"
            : "rc-list-item";

        div.innerHTML = `
            <div class="rc-item__logo-container">
                <img src="${platform.logo}" 
                     alt="${platform.name}" 
                     class="rc-item__logo">
            </div>

            <div class="rc-item__info">
                <div class="rc-item__meta">
                    <span class="rc-item__name">${platform.name}</span>
                    ${index === 0 ? '<span class="rc-item__type">BEST PRICE</span>' : ''}
                </div>
                <div class="rc-item__price">
                    ₹ ${Number(platform.price).toFixed(2)}
                </div>
            </div>

            <div class="rc-item__actions">
                <a href="${platform.url}" target="_blank" class="rc-btn-book">
                    Book Now →
                </a>
            </div>
        `;

        container.appendChild(div);
    });
});

function goBack() {
    sessionStorage.removeItem("fareResults");
    window.location.href = "index.html";
}