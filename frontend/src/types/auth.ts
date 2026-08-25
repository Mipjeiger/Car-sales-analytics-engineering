export interface User {
  id?: string | number;
  name?: string;
  email: string;
  role: UserRole;
}

export type UserRole = 'admin' | 'user' | 'moderator' | 'viewer';