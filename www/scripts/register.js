(function(){
  // Script específico da página de registro
  // Define preferredMode como 'register' para api.js enviar direto para /register
  try {
    if (window._FaceAuth) {
      window._FaceAuth.preferredMode = 'register';
    } else {
      window._FaceAuth = { preferredMode: 'register' };
    }
  } catch (e) { /* ignora erros */ }

  // api.js já cuida do form submit, aqui só focamos o campo de usuário
  const uinp = document.getElementById('input_usuario_reg');
  if (uinp) uinp.focus();
})();
