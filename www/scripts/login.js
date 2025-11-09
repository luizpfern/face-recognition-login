(function(){
  // Script específico da página de login
  // Limpa preferredMode para mostrar confirmação de registro/login e
  // garante que o link "Criar conta" aponta para criar-conta.html
  try {
    if (window._FaceAuth) {
      window._FaceAuth.preferredMode = null;
    } else {
      window._FaceAuth = { preferredMode: null };
    }
  } catch (e) { /* ignora erros */ }

  // UX: garante navegação do link criar conta
  const createLink = document.getElementById('criar_conta');
  if (createLink) {
    createLink.addEventListener('click', function(e){
      // comportamento padrão já aponta para criar-conta.html
    });
  }
})();
