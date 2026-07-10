// =========================================================================
// ESTGEL 3D Visualisation Dashboard - Core Logic
// =========================================================================

let embryoData = null;
let explainabilityReport = null;

let currentPhenotype = "wt"; // "wt" or "rnai"
let currentTimeStep = 30;

// Three.js global variables
let scene, camera, renderer, controls;
let cellGroup, edgeGroup;

// Chart.js global variables
let growthChart = null;
let attentionChart = null;
let selectedInteraction = "ABalaaaal->ABalaaaar";

// Helper function to color code cell lineages
function getLineageColor(cellName) {
    if (cellName.startsWith("AB")) {
        return 0x00f2fe; // Cyan
    } else if (cellName.startsWith("MS")) {
        return 0xffd200; // Yellow
    } else if (cellName.startsWith("E")) {
        return 0x00e676; // Green
    } else if (cellName.startsWith("C")) {
        return 0xff9100; // Orange
    } else if (cellName.startsWith("D")) {
        return 0xf50057; // Pink
    } else if (cellName.startsWith("P")) {
        return 0xff1744; // Red
    }
    return 0xa0aec0; // Gray
}

// Helper to calculate average attention weight
function getAverageAttention(attentions) {
    const vals = Object.values(attentions);
    if (vals.length === 0) return 0.0;
    const sum = vals.reduce((a, b) => a + b, 0);
    return sum / vals.length;
}

// Set up Three.js 3D Viewport
function init3D() {
    const container = document.getElementById("three-canvas-container");
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080a0f);

    camera = new THREE.PerspectiveCamera(45, width / height, 1, 1000);
    // Align camera viewport dynamically
    camera.position.set(100, 100, 300);

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    container.appendChild(renderer.domElement);

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.maxPolarAngle = Math.PI;

    // Add lighting
    const ambientLight = new THREE.AmbientLight(0x222233);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 1.2);
    dirLight.position.set(200, 400, 300);
    scene.add(dirLight);

    const pointLight = new THREE.PointLight(0xffffff, 0.8, 500);
    pointLight.position.set(0, 0, 0);
    camera.add(pointLight);
    scene.add(camera);

    // Add Coordinate Grid Floor
    const gridHelper = new THREE.GridHelper(400, 40, 0x4a5568, 0x1a202c);
    gridHelper.position.y = -80;
    scene.add(gridHelper);

    cellGroup = new THREE.Group();
    edgeGroup = new THREE.Group();
    scene.add(cellGroup);
    scene.add(edgeGroup);

    // Handle window resize
    window.addEventListener("resize", () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });

    animate();
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

// Render the 3D cell nodes and attention lines for a given timestep
function update3DView() {
    if (!embryoData || !explainabilityReport) return;

    // 1. Clear previous meshes
    while (cellGroup.children.length > 0) {
        const obj = cellGroup.children[0];
        obj.geometry.dispose();
        obj.material.dispose();
        cellGroup.remove(obj);
    }
    while (edgeGroup.children.length > 0) {
        const obj = edgeGroup.children[0];
        obj.geometry.dispose();
        obj.material.dispose();
        edgeGroup.remove(obj);
    }

    const typeKey = currentPhenotype; // "wt" or "rnai"
    const tsKey = String(currentTimeStep);
    const stepCells = embryoData[typeKey].timesteps[tsKey] || [];

    // Map to quickly look up cell coordinates in 3D
    const cellPosMap = {};

    // Center of coordinates calculations for camera target focus
    let cx = 0, cy = 0, cz = 0;

    // 2. Draw Cell Nuclei Spheres
    stepCells.forEach(cell => {
        const [x, y, z] = cell.pos;
        cx += x;
        cy += y;
        cz += z;

        cellPosMap[cell.name] = new THREE.Vector3(x, y, z);

        // Normalize and scale visual size
        const radius = Math.max(3, cell.size * 0.4);
        const geometry = new THREE.SphereGeometry(radius, 16, 16);
        
        // Color based on cell lineage (AB, MS, E, C, D)
        const hexColor = getLineageColor(cell.name);
        const material = new THREE.MeshPhongMaterial({
            color: hexColor,
            shininess: 80,
            specular: 0x333333
        });

        const sphere = new THREE.Mesh(geometry, material);
        sphere.position.set(x, y, z);
        cellGroup.add(sphere);
    });

    // Reposition OrbitControls target to center of active cell mass
    if (stepCells.length > 0) {
        controls.target.set(cx / stepCells.length, cy / stepCells.length, cz / stepCells.length);
    }

    // 3. Draw Explainability Attention Edges
    const reportKey = currentPhenotype + "_embryo";
    const stepAttentions = explainabilityReport[reportKey].data.find(
        d => d.timestep === currentTimeStep
    );

    let activeEdgesCount = 0;
    let avgAttentionVal = 0.0;

    if (stepAttentions && stepAttentions.attentions) {
        const attentions = stepAttentions.attentions;
        activeEdgesCount = Object.keys(attentions).length;
        avgAttentionVal = getAverageAttention(attentions);

        const edgeColor = currentPhenotype === "wt" ? 0x00f2fe : 0xff0844;

        Object.entries(attentions).forEach(([connection, weight]) => {
            const [srcName, dstName] = connection.split("->");
            const srcPos = cellPosMap[srcName];
            const dstPos = cellPosMap[dstName];

            if (srcPos && dstPos) {
                // Line weight maps to GNN attention weight
                const material = new THREE.LineBasicMaterial({
                    color: edgeColor,
                    linewidth: weight * 3,
                    transparent: true,
                    opacity: Math.max(0.15, weight)
                });

                const points = [srcPos, dstPos];
                const geometry = new THREE.BufferGeometry().setFromPoints(points);
                const line = new THREE.Line(geometry, material);
                edgeGroup.add(line);
            }
        });

        // Update Top-5 sidebar interactions list
        updateSidebarList(attentions);
    } else {
        updateSidebarList({});
    }

    // Update Stats Display panel
    document.getElementById("stat-nodes").innerText = stepCells.length;
    document.getElementById("stat-edges").innerText = activeEdgesCount;
    document.getElementById("stat-avg-att").innerText = avgAttentionVal.toFixed(3);
}

// Render Top-5 GNN attention connections in the control panel
function updateSidebarList(attentions) {
    const list = document.getElementById("interaction-list");
    list.innerHTML = "";

    const sortedAtts = Object.entries(attentions)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 5);

    if (sortedAtts.length === 0) {
        list.innerHTML = `<li class="interaction-item" style="color: #718096; justify-content: center;">No active connections at t=${currentTimeStep}</li>`;
        return;
    }

    sortedAtts.forEach(([connection, weight]) => {
        const li = document.createElement("li");
        li.className = "interaction-item";
        li.style.cursor = "pointer";
        li.innerHTML = `
            <span class="interaction-name">${connection}</span>
            <span class="interaction-weight">${weight.toFixed(4)}</span>
        `;
        
        // Clicking on an interaction details updates the Attention Trajectory Contrast Chart
        li.addEventListener("click", () => {
            selectedInteraction = connection;
            updateAttentionChart();
        });
        
        list.appendChild(li);
    });
}

// Chart 1: Active Cell Growth curve using Chart.js
function initGrowthChart() {
    const ctx = document.getElementById("growthChart").getContext("2d");
    
    // Prepare time steps and counts
    const steps = [30, 60, 90, 120, 150, 180];
    const wtCounts = steps.map(t => (embryoData.wt.timesteps[String(t)] || []).length);
    const rnaiCounts = steps.map(t => (embryoData.rnai.timesteps[String(t)] || []).length);

    growthChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: steps.map(t => `t=${t}`),
            datasets: [
                {
                    label: "Control (WT)",
                    data: wtCounts,
                    borderColor: "#4facfe",
                    backgroundColor: "rgba(79, 172, 254, 0.1)",
                    borderWidth: 2,
                    tension: 0.2,
                    fill: true
                },
                {
                    label: "Perturbed (RNAi)",
                    data: rnaiCounts,
                    borderColor: "#ff0844",
                    backgroundColor: "rgba(255, 8, 68, 0.1)",
                    borderWidth: 2,
                    borderDash: [4, 4],
                    tension: 0.2,
                    fill: true
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#a0aec0", font: { family: "Outfit" } } }
            },
            scales: {
                x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#718096" } },
                y: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#718096" } }
            }
        }
    });
}

// Chart 2: EAM Attention Trajectory Contrast (WT vs. RNAi)
function updateAttentionChart() {
    const ctx = document.getElementById("attentionChart").getContext("2d");

    const steps = [30, 60, 90, 120, 150, 180];
    
    // Extract trajectories for WT
    const wtVals = steps.map(t => {
        const step = explainabilityReport.wt_embryo.data.find(d => d.timestep === t);
        return step ? (step.attentions[selectedInteraction] || 0.0) : 0.0;
    });

    // Extract trajectories for RNAi
    const rnaiVals = steps.map(t => {
        const step = explainabilityReport.rnai_embryo.data.find(d => d.timestep === t);
        return step ? (step.attentions[selectedInteraction] || 0.0) : 0.0;
    });

    if (attentionChart) {
        attentionChart.destroy();
    }

    attentionChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: steps.map(t => `t=${t}`),
            datasets: [
                {
                    label: `WT: ${selectedInteraction}`,
                    data: wtVals,
                    borderColor: "#00f2fe",
                    backgroundColor: "#00f2fe",
                    borderWidth: 3,
                    pointRadius: 4,
                    tension: 0.1
                },
                {
                    label: `RNAi: ${selectedInteraction}`,
                    data: rnaiVals,
                    borderColor: "#ff0844",
                    backgroundColor: "#ff0844",
                    borderWidth: 3,
                    borderDash: [5, 5],
                    pointRadius: 4,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: "#a0aec0", font: { family: "Outfit" } } }
            },
            scales: {
                x: { grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#718096" } },
                y: { min: 0.0, max: 1.0, grid: { color: "rgba(255,255,255,0.05)" }, ticks: { color: "#718096" } }
            }
        }
    });
}

// Fetch JSON data and initialize elements
async function initDashboard() {
    try {
        console.log("Loading dashboard data...");
        const [embryoRes, explainRes] = await Promise.all([
            fetch("embryo_data.json"),
            fetch("explainability_report.json")
        ]);

        embryoData = await embryoRes.json();
        explainabilityReport = await explainRes.json();

        console.log("Data loaded successfully.");

        // Initialize visualization components
        init3D();
        initGrowthChart();
        updateAttentionChart();
        update3DView();

        // Setup HTML UI Event Listeners
        const btnWt = document.getElementById("btn-wt");
        const btnRnai = document.getElementById("btn-rnai");
        const slider = document.getElementById("time-slider");
        const timeVal = document.getElementById("current-time-val");

        btnWt.addEventListener("click", () => {
            btnWt.classList.add("active");
            btnRnai.classList.remove("active");
            currentPhenotype = "wt";
            update3DView();
        });

        btnRnai.addEventListener("click", () => {
            btnRnai.classList.add("active");
            btnWt.classList.remove("active");
            currentPhenotype = "rnai";
            update3DView();
        });

        slider.addEventListener("input", (e) => {
            currentTimeStep = parseInt(e.target.value);
            timeVal.innerText = currentTimeStep;
            update3DView();
        });

    } catch (err) {
        console.error("Error loading dashboard data:", err);
        document.body.innerHTML = `<div style="display:flex; justify-content:center; align-items:center; height:100vh; color:red; font-size:18px;">
            Error loading JSON data. Did you run the exporter script (python scripts/export_ui_data.py)?
        </div>`;
    }
}

window.addEventListener("DOMContentLoaded", initDashboard);
