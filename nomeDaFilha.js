const nomeDaFilha = (nomeCompletoDaMae, nomeCompletoDoPai, nomeDaFilha) => {
  // 'Rafael Santos Fischer' -> ['Rafael', 'Santos', 'Fischer'] -> 'Fischer'
  const tamanhoDoNomeCompletoDoPai = nomeCompletoDoPai.split(" ").length;
  const sobreNomeDoPai =
    nomeCompletoDoPai.split(" ")[tamanhoDoNomeCompletoDoPai - 1];
  const sobreNomeDaMae = nomeCompletoDaMae.split(" ")[1];
  const nomeCompletoDaFilha = `${nomeDaFilha} ${sobreNomeDaMae} ${sobreNomeDoPai}`;
  return nomeCompletoDaFilha;
};
