import { useMutation, useQuery } from "@tanstack/react-query";
import { chatApi } from "@/api/endpoints/chat";
import { useAppDispatch, useAppSelector } from "@/store/hooks";
import { addMessage, resetChat, setEntities, setSessionId, setTyping } from "@/store/slices/chatSlice";

export function useChat() {
  const dispatch = useAppDispatch();
  const { messages, sessionId, entities, isTyping } = useAppSelector((s) => s.chat);

  const intentsQuery = useQuery({
    queryKey: ["chat-intents"],
    queryFn: chatApi.intents,
    staleTime: 5 * 60_000,
  });

  const sendMutation = useMutation({
    mutationFn: (message: string) => chatApi.send({ message, session_id: sessionId }),
    onMutate: (message) => {
      dispatch(
        addMessage({
          id: crypto.randomUUID(),
          role: "user",
          content: message,
          timestamp: new Date().toISOString(),
        }),
      );
      dispatch(setTyping(true));
    },
    onSuccess: (data) => {
      dispatch(setSessionId(data.session_id));
      dispatch(setEntities(data.entities ?? {}));
      dispatch(
        addMessage({
          id: crypto.randomUUID(),
          role: "assistant",
          content: data.response,
          intent: data.intent,
          timestamp: data.timestamp,
        }),
      );
    },
    onSettled: () => dispatch(setTyping(false)),
  });

  const resetMutation = useMutation({
    mutationFn: () => chatApi.reset(sessionId),
    onSuccess: () => dispatch(resetChat()),
    onError: () => dispatch(resetChat()),
  });

  return {
    messages,
    entities,
    isTyping,
    intents: intentsQuery.data?.intents ?? [],
    send: (message: string) => sendMutation.mutateAsync(message),
    reset: () => resetMutation.mutateAsync(),
    isSending: sendMutation.isLoading,
    error: sendMutation.error instanceof Error ? sendMutation.error.message : null,
  };
}
