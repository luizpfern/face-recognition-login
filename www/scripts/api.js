/**
 * scripts/api.js
 * Comportamento para fluxo de login/registro por reconhecimento facial.
 *
 * Funcionalidades:
 * - Intercepta o submit do formulário.
 * - Abre a câmera (getUserMedia) e exibe modal/overlay com preview.
 * - Captura 3 fotos (frente e laterais com instruções).
 * - Envia imagens como multipart/form-data para /register ou /login.
 * - Trata respostas e mostra mensagens status/erro.
 *
 * Observações:
 * - Endpoints: POST /register e POST /login (multipart: username + images[]).
 * - A API retorna códigos de erro amigáveis:
 *     "no_face_detected" - nenhum rosto detectado
 *     "multiple_faces" - múltiplas faces na foto
 *     "face_blurry" - foto borrada
 *     "face_dark" - foto escura
 *     "face_too_small" - rosto muito pequeno/distante
 *     "encoding_failed" - falha ao extrair features
 *     "user_not_found" - usuário não encontrado
 *     "face_already_registered" - rosto já cadastrado
 */

(function () {
  // Configurações
  const CAPTURE_COUNT = 3;
  const CAPTURE_DELAY_MS = 400; // tempo entre capturas para permitir virar a cabeça
  const IMAGE_QUALITY = 0.75; // JPEG quality
  const MAX_WIDTH = 640; // canvas max width
  const MAX_HEIGHT = 480;

  // Verifica se tem câmera disponível
  async function checkCameraAvailability() {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      return devices.some(device => device.kind === 'videoinput');
    } catch (err) {
      console.error('Erro ao verificar câmeras:', err);
      return false;
    }
  }

  // Elementos do DOM (busca robusta para aceitar login ou registro em páginas diferentes)
  const form = document.querySelector('#form-login, #form-register');
  const usernameInput = document.querySelector('#input_usuario, #input_usuario_reg, input[name="username"], input[name="usuario"]');
  const entrarBtn = document.getElementById('entrar');

  // Cria elementos dinâmicos usados durante captura
  let overlay = null;
  let video = null;
  let startCamBtn = null;
  let captureBtn = null;
  let closeBtn = null;
  let statusEl = null;
  let stream = null;

  // Helper: cria o overlay/modal com video preview e controles
  function createOverlay() {
    overlay = document.createElement('div');
    overlay.id = 'fr-modal-overlay';
    Object.assign(overlay.style, {
      position: 'fixed',
      inset: '0',
      background: 'rgba(0,0,0,0.6)',
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      zIndex: 9999,
    });

    const box = document.createElement('div');
    Object.assign(box.style, {
      width: '740px',
      maxWidth: '95%',
      background: '#fff',
      padding: '16px',
      borderRadius: '8px',
      textAlign: 'center',
    });

    const title = document.createElement('h2');
    title.innerText = 'Captura Facial';
    box.appendChild(title);

    const instructions = document.createElement('p');
    instructions.id = 'fr-instructions';
    instructions.innerText = 'Centralize o rosto e clique em "Iniciar câmera". Depois siga as instruções para as 3 fotos (frente e laterais).';
    box.appendChild(instructions);

    video = document.createElement('video');
    video.autoplay = true;
    video.playsInline = true;
    video.width = 640;
    video.height = 480;
    Object.assign(video.style, { border: '1px solid #ccc', borderRadius: '4px' });
    box.appendChild(video);

    statusEl = document.createElement('div');
    statusEl.id = 'fr-status';
    statusEl.style.margin = '8px 0';
    box.appendChild(statusEl);

    const controls = document.createElement('div');
    controls.style.marginTop = '8px';

    startCamBtn = document.createElement('button');
    startCamBtn.type = 'button';
    startCamBtn.innerHTML = '<span>Iniciar câmera</span>';
    startCamBtn.className = 'btn';
    startCamBtn.style.marginRight = '8px';
    startCamBtn.style.minWidth = '140px';
    controls.appendChild(startCamBtn);

    captureBtn = document.createElement('button');
    captureBtn.type = 'button';
    captureBtn.innerHTML = '<span>Capturar fotos</span>';
    captureBtn.className = 'btn';
    captureBtn.disabled = true;
    captureBtn.style.marginRight = '8px';
    captureBtn.style.minWidth = '140px';
    controls.appendChild(captureBtn);

    closeBtn = document.createElement('button');
    closeBtn.type = 'button';
    closeBtn.innerText = 'Cancelar';
    closeBtn.className = 'btn-secondary';
    controls.appendChild(closeBtn);

    box.appendChild(controls);
    overlay.appendChild(box);
    document.body.appendChild(overlay);

    // Event listeners
    startCamBtn.addEventListener('click', startCamera);
    captureBtn.addEventListener('click', onCaptureClicked);
    closeBtn.addEventListener('click', cleanupAndCloseOverlay);
  }

  function showStatus(text, type = 'info') {
    if (!statusEl) return;
    statusEl.innerText = text;
    
    // Estilos por tipo de mensagem
    const styles = {
      info: { bg: '#f3f4f6', color: '#374151' },
      success: { bg: '#dcfce7', color: '#166534' },
      error: { bg: '#fee2e2', color: '#991b1b' },
      warning: { bg: '#fff7ed', color: '#9a3412' }
    };
    
    Object.assign(statusEl.style, {
      backgroundColor: styles[type]?.bg || styles.info.bg,
      color: styles[type]?.color || styles.info.color,
      padding: '12px',
      borderRadius: '6px',
      marginTop: '12px',
      marginBottom: '12px'
    });
  }

  // Helper para mostrar loading state nos botões
  function setButtonLoading(button, isLoading, originalText) {
    if (!button) return;
    if (isLoading) {
      button.disabled = true;
      button.innerHTML = '<span class="loading"></span><span>Aguarde...</span>';
    } else {
      button.disabled = false;
      button.innerHTML = `<span>${originalText}</span>`;
    }
  }

  async function startCamera() {
    showStatus('Solicitando permissão para câmera...', 'info');
    setButtonLoading(startCamBtn, true);

    try {
      stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          facingMode: 'user',
          width: { ideal: MAX_WIDTH },
          height: { ideal: MAX_HEIGHT },
        }, 
        audio: false 
      });
      
      video.srcObject = stream;
      video.onloadedmetadata = () => {
        captureBtn.disabled = false;
        startCamBtn.disabled = true;
        showStatus('Câmera ativa. Centralize seu rosto e mantenha boa iluminação.', 'success');
        setButtonLoading(startCamBtn, false, 'Iniciar câmera');
        
        // Inicia análise em tempo real
        //startQualityCheck();
      };

    } catch (err) {
      console.error('Erro ao iniciar câmera:', err);
      let errorMsg = 'Erro ao acessar câmera: ';
      
      if (err.name === 'NotAllowedError' || err.name === 'PermissionDeniedError') {
        errorMsg += 'Permissão negada. Por favor, permita o acesso à câmera.';
      } else if (err.name === 'NotFoundError' || err.name === 'DevicesNotFoundError') {
        errorMsg += 'Nenhuma câmera encontrada.';
      } else if (err.name === 'NotReadableError' || err.name === 'TrackStartError') {
        errorMsg += 'Câmera em uso por outro aplicativo.';
      } else {
        errorMsg += err.message || 'Verifique se sua câmera está conectada e funcionando.';
      }
      
      showStatus(errorMsg, 'error');
      setButtonLoading(startCamBtn, false, 'Iniciar câmera');
    }
  }

  async function stopCamera() {
    if (stream) {
      const tracks = stream.getTracks();
      tracks.forEach(t => t.stop());
      stream = null;
    }
    if (video) {
      try {
        video.srcObject = null;
      } catch (e) {}
    }
  }

  function cleanupAndCloseOverlay() {
    stopCamera();
    if (overlay && overlay.parentNode) overlay.parentNode.removeChild(overlay);
    overlay = null;
    video = null;
    startCamBtn = null;
    captureBtn = null;
    closeBtn = null;
    statusEl = null;
  }

  // Captura n frames com delay entre elas e retorna array de Blobs (JPEG)
  async function captureFrames(n = CAPTURE_COUNT, delay = CAPTURE_DELAY_MS) {
    if (!video) throw new Error('Video element não inicializado');

    const canvas = document.createElement('canvas');
    // dimensionar canvas baseado no vídeo com limite de MAX_WIDTH/MAX_HEIGHT
    const vw = video.videoWidth || 640;
    const vh = video.videoHeight || 480;
    let scale = Math.min(1, MAX_WIDTH / vw, MAX_HEIGHT / vh);
    canvas.width = Math.round(vw * scale);
    canvas.height = Math.round(vh * scale);
    const ctx = canvas.getContext('2d');

    const blobs = [];
    for (let i = 0; i < n; i++) {
      // instruções para o usuário entre capturas
      if (i === 0) {
        showStatus('Caputrando imagens... 1/3');
      } else if (i === 1) {
        showStatus('Caputrando imagens... 2/3');
      } else {
        showStatus('Caputrando imagens... 3/3');
      }

      // esperar um pequeno tempo antes de capturar (para movimento)
      await new Promise(r => setTimeout(r, 600));

      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const blob = await new Promise(res => canvas.toBlob(res, 'image/jpeg', IMAGE_QUALITY));
      blobs.push(blob);

      // intervalo entre capturas
      await new Promise(r => setTimeout(r, delay));
    }
    showStatus('Capturas realizadas.');
    return blobs;
  }

  // Envia imagens ao endpoint ("/login" ou "/register")
  async function sendImages(endpoint, username, blobs) {
    showStatus('Preparando envio...');
    const form = new FormData();
    form.append('username', username);
    blobs.forEach((b, i) => form.append('images', b, `img_${i}.jpg`));

    showStatus('Enviando imagens para a API...');

    console.log('Enviando para endpoint:', endpoint);
    console.log('Dados a serem enviados:', Array.from(form.entries()));

    try {
      const resp = await fetch(endpoint, {
        method: 'POST',
        body: form,
      });
      const json = await resp.json();
      handleApiResponse(json, endpoint);
    } catch (err) {
      console.error('Erro no envio:', err);
      showStatus('Erro ao enviar: ' + (err.message || err));
    }
  }

  function handleApiResponse(json, endpoint) {
    console.log("🚀 ~ handleApiResponse ~ endpoint:", endpoint)
    console.log("🚀 ~ handleApiResponse ~ json1:", json)
    // Tratar responses padrão e reason codes
    if (!json) {
      showStatus('Resposta vazia do servidor.');
      return;
    }

    if (endpoint === '/register') {
      if (json.success === true) {
        alertModal('Registro efetuado com sucesso.');
        return;
      } else {
        const reason = json.reason || json.message || (json.success === false ? 'failed' : null);
        if (reason) {
          const friendly = mapReasonToFriendly(reason);
          alertModal(friendly);
          return;
        }
      }
    } else if (endpoint === '/login') {
      // Redirecionar para tela principal de autenticado!
      if (json.authenticated === true) {
        alertModal('Autenticado com sucesso! Votes: ' + (json.votes ?? '?') + ', avg_distance: ' + (json.avg_distance ?? '?'))
          .then(() => {
            window.location.href = 'pages/success.html'; // ou outra página principal
          });
        return;
      } else {
        const reason = json.reason || json.message || (json.authenticated === false ? 'not_matched' : null);
        alertModal(mapReasonToFriendly(reason));
        return;
      }
    }
  }

  function mapReasonToFriendly(reason) {
    // Mapeia codes do backend para mensagens amigáveis ao usuário
    const map = {
      no_face_detected: 'Nenhum rosto detectado. Centralize seu rosto e tente novamente.',
      multiple_faces: 'Foram detectadas múltiplas faces. Certifique-se de estar sozinho no enquadramento.',
      face_blurry: 'Imagem borrada. Mantenha a câmera estável e tente de novo.',
      face_dark: 'Imagem escura. Aumente a iluminação.',
      face_overexposed: 'Imagem superexposta. Ajuste a iluminação.',
      face_too_small: 'Rosto muito distante/pequeno. Aproxime-se da câmera.',
      encoding_failed: 'Não foi possível extrair características do rosto. Tente outra foto.',
      user_not_found: 'Usuário não encontrado. Verifique o nome ou registre-se primeiro.',
      face_already_registered: 'Esse rosto já está registrado. Caso seja você, faça login.',
      not_matched: 'Rosto não corresponde ao usuário informado.',
      invalid_image: 'Imagem inválida.',
      failed: 'Operação falhou. Tente novamente.',
    };
    return map[reason] || reason;
  }

  // Handler quando usuário clica em "Seguir para a Câmera" (form submit)
  if (form) {
    form.addEventListener('submit', async function (e) {
      e.preventDefault();
      const username = usernameInput && usernameInput.value ? usernameInput.value.trim() : '';
      if (!username) {
        alert('Informe o usuário antes de prosseguir.');
        return;
      }

      // Criar overlay/modal de captura
      if (!overlay) createOverlay();

      // Abrir câmera se ainda não aberta (startCamera será chamada pelo overlay)
      // Ativa instrução para o usuário: depois de iniciar a câmera, clicar em "Capturar 3 fotos".
      showStatus('Abra a câmera e capture 3 fotos para continuar.');

      // Observação: não iniciamos a captura automaticamente para dar controle ao usuário.
    });
  }

  // Função chamada quando o botão "Capturar 3 fotos" no overlay é pressionado
  async function onCaptureClicked() {
    const username = usernameInput.value.trim();
    if (!username) {
      showStatus('Informe o usuário antes de capturar.');
      return;
    }
    try {
      // Captura as imagens
      const blobs = await captureFrames(CAPTURE_COUNT, CAPTURE_DELAY_MS);

      // Desliga a câmera imediatamente após capturar
      await stopCamera();
      startCamBtn.disabled = false; // permite reiniciar se precisar
      captureBtn.disabled = true;

      // Decidir endpoint. Permite que outros scripts prefiram o modo (ex: register.js define preferredMode)
      const preferred = window._FaceAuth && window._FaceAuth.preferredMode ? window._FaceAuth.preferredMode : null;
      const isRegister = preferred === 'register' ? true : false;
      const endpoint = isRegister ? '/register' : '/login';
      await sendImages(endpoint, username, blobs);

      // Fechar overlay após envio (mantemos por 2s para leitura)
      setTimeout(() => cleanupAndCloseOverlay(), 1500);
    } catch (err) {
      console.error('Erro durante captura/envio:', err);
      showStatus('Erro ao capturar: ' + (err.message || err));
    }
  }

  async function alertModal(message) { // modal de alerta seguindo estilo da página
    return new Promise((resolve) => {
      // Remove modal anterior se existir
      const existingModal = document.getElementById('fr-alert-modal');
      if (existingModal) existingModal.remove();

      // Criar overlay
      const alertOverlay = document.createElement('div');
      alertOverlay.id = 'fr-alert-modal';
      Object.assign(alertOverlay.style, {
        position: 'fixed',
        inset: '0',
        background: 'rgba(0,0,0,0.6)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 10000,
        animation: 'fadeIn 0.2s ease-out'
      });

      // Criar caixa do modal
      const modalBox = document.createElement('div');
      Object.assign(modalBox.style, {
        width: '400px',
        maxWidth: '90%',
        background: '#fff',
        padding: '24px',
        borderRadius: '8px',
        boxShadow: '0 4px 6px rgba(0,0,0,0.1)',
        animation: 'slideIn 0.3s ease-out'
      });

      // Título do modal
      const title = document.createElement('h3');
      title.innerText = 'Atenção';
      Object.assign(title.style, {
        margin: '0 0 16px 0',
        fontSize: '20px',
        fontWeight: '600',
        color: '#374151'
      });
      modalBox.appendChild(title);

      // Mensagem
      const messageEl = document.createElement('p');
      messageEl.innerText = message;
      Object.assign(messageEl.style, {
        margin: '0 0 24px 0',
        fontSize: '15px',
        lineHeight: '1.5',
        color: '#6b7280'
      });
      modalBox.appendChild(messageEl);

      // Botão OK
      const okBtn = document.createElement('button');
      okBtn.type = 'button';
      okBtn.innerText = 'OK';
      okBtn.className = 'btn';
      Object.assign(okBtn.style, {
        width: '100%',
        padding: '10px',
        fontSize: '15px',
        cursor: 'pointer'
      });
      
      okBtn.addEventListener('click', () => {
        alertOverlay.remove();
        resolve();
      });
      
      modalBox.appendChild(okBtn);
      alertOverlay.appendChild(modalBox);
      document.body.appendChild(alertOverlay);

      // Adicionar animações CSS
      const style = document.createElement('style');
      style.textContent = `
        @keyframes fadeIn {
          from { opacity: 0; }
          to { opacity: 1; }
        }
        @keyframes slideIn {
          from { transform: translateY(-20px); opacity: 0; }
          to { transform: translateY(0); opacity: 1; }
        }
      `;
      document.head.appendChild(style);
    });
  }

  // Expõe funções úteis para depuração (opcional) e allow preferred mode control
  window._FaceAuth = {
    startCamera,
    stopCamera,
    captureFrames,
    sendImages,
    preferredMode: null,
    setPreferredMode(mode){
      this.preferredMode = mode; 
    }
  };
})();