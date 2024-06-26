defmodule Pureai.AzureEmbedding do
  use HTTPoison.Base

  def api_key() do
    Application.get_env(:openai, :azure_openai_api_key)
  end

  def api_deployment() do
    Application.get_env(:openai, :azure_openai_embeddings_deployment)
  end

  def api_ver() do
    Application.get_env(:openai, :azure_openai_api_version)
  end

  def client() do
    Tesla.client([
      {Tesla.Middleware.BaseUrl, Application.get_env(:openai, :azure_openai_endpoint)},
      Tesla.Middleware.JSON,
      {Tesla.Middleware.Headers, [{"api-key", "#{api_key()}"}]}
    ])
  end

  def text_to_vetor(prompt) do
    client() |> text_to_vetor(prompt)
  end

  def text_to_vetor(client, prompt) do
    sha = hash_text(prompt)

    try do
      case PureAI.Context.get_embedding_vector_by_sha(sha) do
        nil ->
          mp = %{:input => prompt}
          {:ok, %Tesla.Env{status: status, body: %{"data" => [%{"embedding" => vectors}]}}} = do_text_to_vector(client, mp, :openai)
          Tesla.post(client, "/openai/deployments/#{api_deployment()}/embeddings?api-version=#{api_ver()}", mp)

          case status do
            200 ->
              PureAI.Context.create_embedding_vector(%{sha: sha, text: prompt, vector: Jason.encode!(vectors)})

            _ ->
              {:error, status}
          end

        %{vector: _} = res ->
          {:ok, res}
      end
    rescue
      error ->
        {:error, inspect(error)}
    end
  end

  def do_text_to_vector(client, mp, :openai) do
    {:ok, %Tesla.Env{status: status, body: %{"data" => [%{"embedding" => vectors}]}}} =
      Tesla.post(
        client,
        "/openai/deployments/#{api_deployment()}/embeddings?api-version=#{api_ver()}",
        mp
      )
  end

  def hash_text(text) do
    # binary_text = text |> String.to_charlist() |> :erlang.iolist_to_binary()
    hashed_binary = :crypto.hash(:sha256, text)
    Base.encode16(hashed_binary, case: :lower)
  end
end
