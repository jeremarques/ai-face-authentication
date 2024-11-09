const videoCam = document.getElementById('video-cam');

// Pré-carrega modelos para que a função `startDetection` seja executada mais rapidamente.
const loadModels = async () => {
    await Promise.all([
        faceapi.nets.tinyFaceDetector.loadFromUri('./models'),
        faceapi.nets.faceRecognitionNet.loadFromUri('./models'),
        faceapi.nets.faceLandmark68Net.loadFromUri('./models'),
        faceapi.nets.ssdMobilenetv1.loadFromUri('./models'),
    ]);
};

// Inicializa o streaming de vídeo e canvas
const initializeVideoStream = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 320, height: 240 } // Resolução reduzida para melhor desempenho
    });
    videoCam.srcObject = stream;
};

// Configuração do Canvas
let canvas;
const setupCanvas = () => {
    if (!canvas) {
        canvas = faceapi.createCanvasFromMedia(videoCam);
        document.body.append(canvas);
        const displaySize = { width: videoCam.width, height: videoCam.height };
        faceapi.matchDimensions(canvas, displaySize);
    }
};

// Função para iniciar a detecção
const startDetection = async () => {
    await initializeVideoStream();
    await loadModels();
    setupCanvas();

    const displaySize = { width: videoCam.width, height: videoCam.height };
    const labeledFaceDescriptors = await loadLabeledImages();
    const faceMatcher = new faceapi.FaceMatcher(labeledFaceDescriptors, 0.6);

    // Detecção otimizada usando `requestAnimationFrame`
    const detectFaces = async (roi) => {
        const faceAIData = await faceapi
            .detectAllFaces(videoCam, new faceapi.TinyFaceDetectorOptions(), roi)
            .withFaceLandmarks()
            .withFaceDescriptors();

        // Limpeza e redimensionamento de resultados
        const resizedResults = faceapi.resizeResults(faceAIData, displaySize);
        const context = canvas.getContext('2d');
        context.clearRect(0, 0, canvas.width, canvas.height);

        const results = resizedResults.map(d => faceMatcher.findBestMatch(d.descriptor));
        results.forEach((result, i) => {
            const box = resizedResults[i].detection.box;
            const drawBox = new faceapi.draw.DrawBox(box, { label: result.toString() });
            drawBox.draw(canvas);
        })

        // Solicita próxima detecção
        requestAnimationFrame(detectFaces);
    };

    // Inicia o loop de detecção
    detectFaces();
};

function loadLabeledImages() {
    const labels = ['Jeremias'];
    return Promise.all(
        labels.map(async label => {
            const descriptions = [];
            for (let i = 1; i <= 2; i++) {
                const img = await faceapi.fetchImage(`https://raw.githubusercontent.com/jeremarques/ai-face-authentication/master/labeled_images/${label}/${i}.jpeg`);
                const detections = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
                descriptions.push(detections.descriptor);
            }
    
            return new faceapi.LabeledFaceDescriptors(label, descriptions);
        })
    )
}

// Chamada inicial para iniciar a detecção
startDetection();
