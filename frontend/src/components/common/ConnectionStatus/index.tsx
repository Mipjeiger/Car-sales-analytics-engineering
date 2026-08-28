// frontend/src/components/common/ConnectionStatus/index.tsx
import { Box, Chip, Tooltip } from '@mui/material';
import { styled } from '@mui/material/styles';

interface ConnectionStatusProps {
  isConnected: boolean;
  error?: string | null;
  showLabel?: boolean;
  size?: 'small' | 'medium';
}

const StatusDot = styled(Box)<{ isconnected: string }>(({ theme, isconnected }) => ({
  width: 10,
  height: 10,
  borderRadius: '50%',
  backgroundColor: isconnected === 'true' ? theme.palette.success.main : theme.palette.error.main,
  display: 'inline-block',
  animation: isconnected === 'true' ? 'pulse 2s infinite' : 'none',
  '@keyframes pulse': {
    '0%': {
      opacity: 1,
    },
    '50%': {
      opacity: 0.5,
    },
    '100%': {
      opacity: 1,
    },
  },
}));

export default function ConnectionStatus({ 
  isConnected, 
  error = null, 
  showLabel = true,
  size = 'medium'
}: ConnectionStatusProps) {
  const statusText = isConnected ? 'Connected' : error || 'Disconnected';
  const statusColor = isConnected ? 'success' : error ? 'warning' : 'error';

  return (
    <Tooltip title={error || (isConnected ? 'Connected to server' : 'Disconnected from server')}>
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <StatusDot isconnected={isConnected.toString()} />
        {showLabel && (
          <Chip
            label={statusText}
            color={statusColor}
            size={size}
            variant="outlined"
            sx={{ 
              fontWeight: 500,
              '& .MuiChip-label': {
                px: 1,
              }
            }}
          />
        )}
      </Box>
    </Tooltip>
  );
}