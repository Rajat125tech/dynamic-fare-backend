document.getElementById("fareForm").addEventListener("submit", function (e) {
    e.preventDefault();
    predictFare();
});

function getUserLocation() {
    return new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                resolve({
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude
                });
            },
            (error) => {
                console.error("Geolocation error:", error);
                reject(error);
            },
            {
                enableHighAccuracy: true,
                timeout: 10000,
                maximumAge: 0
            }
        );
    });
}

async function predictFare() {
    const distance = parseFloat(document.getElementById("distance").value);
    const duration = parseFloat(document.getElementById("duration").value);
    const vehicle_type = document.getElementById("vehicle_type").value;
    const num_passengers = parseInt(document.getElementById("num_passengers").value);

    const btn = document.getElementById("predictBtn");
    btn.disabled = true;
    btn.innerText = "Comparing...";

    let location;

    try {
        location = await getUserLocation();
        console.log("Geolocation:", location);
    } catch (err) {
        alert("Location permission required.");
        btn.disabled = false;
        btn.innerText = "Compare Prices";
        return;
    }

    const payload = {
        distance: distance,
        duration: duration,
        vehicle_type: vehicle_type,
        num_passengers: num_passengers,
        latitude: location.latitude,
        longitude: location.longitude
    };

    console.log("FINAL PAYLOAD:", payload);

    try {
        const response = await fetch("https://dynamic-fare-backend.onrender.com/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (!response.ok) {
            console.error("Backend error:", data);
            alert("Backend validation error. Check console.");
            return;
        }

        sessionStorage.setItem("fareResults", JSON.stringify(data));
        window.location.href = "results.html";

    } catch (error) {
        console.error(error);
        alert("Network error.");
    } finally {
        btn.disabled = false;
        btn.innerText = "Compare Prices";
    }
}